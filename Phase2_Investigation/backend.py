
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import pandas as pd
import tempfile
import os
import shutil
import subprocess
import json
import networkx as nx
from tqdm import tqdm

app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend files (index.html and assets) from the project root.
# This makes the frontend and backend same-origin and avoids `Failed to fetch` CORS/network issues
app.mount("/", StaticFiles(directory='.', html=True), name="static")


# Use the default transaction CSV for graph construction

TRANSACTION_CSV = "HI-Small_Trans.csv"
G = None
TX_DF = None
GRAPH_CACHE = "graph.gpickle"

# Build the graph at server startup
def build_graph_on_startup():
    global G, TX_DF
    print("[Startup] Loading transactions and building graph...")
    # If cached graph exists, load it to speed startup
    if os.path.exists(GRAPH_CACHE):
        print(f"[Startup] Loading cached graph from {GRAPH_CACHE}...")
        G = nx.read_gpickle(GRAPH_CACHE)
        TX_DF = None
        print(f"[Startup] Loaded cached graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        return

    TX_DF = pd.read_csv(TRANSACTION_CSV)
    TX_DF['Timestamp'] = pd.to_datetime(TX_DF['Timestamp'], errors='coerce')
    TX_DF = TX_DF.dropna(subset=['Account', 'Account.1', 'Amount Received'])
    TX_DF['Amount Received'] = pd.to_numeric(TX_DF['Amount Received'], errors='coerce').fillna(0.0)
    TX_DF['Account'] = TX_DF['Account'].astype(str)
    TX_DF['Account.1'] = TX_DF['Account.1'].astype(str)
    G = nx.DiGraph()
    for _, row in tqdm(TX_DF.iterrows(), total=len(TX_DF), desc="[Startup] Adding edges"):
        u = row['Account']
        v = row['Account.1']
        amt = float(row['Amount Received'])
        ts = row['Timestamp']
        txid = row.get('transaction_id', None) or row.name
        if G.has_edge(u, v):
            G[u][v]['tx_ids'].append(txid)
            G[u][v]['tx_count'] += 1
            G[u][v]['total_amount'] += amt
            if ts is not pd.NaT:
                if 'last_tx' not in G[u][v] or pd.isna(G[u][v]['last_tx']) or ts > G[u][v]['last_tx']:
                    G[u][v]['last_tx'] = ts
        else:
            G.add_edge(u, v,
                       tx_ids=[txid],
                       tx_count=1,
                       total_amount=amt,
                       first_tx=ts,
                       last_tx=ts)
    for n in tqdm(G.nodes(), desc="[Startup] Calculating node features"):
        G.nodes[n]['inflow'] = sum(G[p][n]['total_amount'] for p in G.predecessors(n) if G.has_edge(p, n))
        G.nodes[n]['outflow'] = sum(G[n][q]['total_amount'] for q in G.successors(n) if G.has_edge(n, q))
        G.nodes[n]['weighted_degree'] = G.nodes[n]['inflow'] + G.nodes[n]['outflow']
    print(f"[Startup] Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    try:
        nx.write_gpickle(G, GRAPH_CACHE)
        print(f"[Startup] Cached graph saved to {GRAPH_CACHE}")
    except Exception as e:
        print("[Startup] Failed to save graph cache:", e)

@app.on_event("startup")
def on_startup():
    build_graph_on_startup()


@app.post("/investigate/")
async def investigate(
    seed_file: UploadFile | None = File(None),
    suspicious_accounts: str | None = Form(None),
    k_hop: int = Form(...)
):
    # Determine seed accounts: either from uploaded CSV or from the text field
    seed_accounts = set()
    if suspicious_accounts:
        # parse comma or newline separated list
        parts = [s.strip() for s in suspicious_accounts.replace('\r','').replace('\n',',').split(',') if s.strip()]
        seed_accounts = set(parts)
    elif seed_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = os.path.join(tmpdir, seed_file.filename)
            with open(seed_path, "wb") as f:
                shutil.copyfileobj(seed_file.file, f)
            seeds_df = pd.read_csv(seed_path)
            seed_accounts = set(seeds_df.iloc[:, 0].astype(str))
    else:
        return JSONResponse({"error": "No seed accounts provided."}, status_code=400)

    # Determine which seeds are valid and which are not
    valid_seeds = {n for n in seed_accounts if n in G}
    invalid_seeds = [n for n in seed_accounts if n not in G]
    if not valid_seeds:
        return JSONResponse({"error": "No valid seed accounts found in the graph.", "invalid_seeds": invalid_seeds}, status_code=400)

    # Now run the investigation using the pre-built G and these seeds
    # --- Investigation logic (simplified, you can expand as needed) ---
    # k-hop expansion
    def k_hop_neighbors(G, seeds, k=2):
        R = set(seeds)
        frontier = set(seeds)
        for _ in range(k):
            nxt = set()
            for u in frontier:
                nxt |= set(G.predecessors(u)) if u in G else set()
                nxt |= set(G.successors(u)) if u in G else set()
            nxt -= R
            R |= nxt
            frontier = nxt
            if not frontier:
                break
        return R

    k2_neighborhood = k_hop_neighbors(G, valid_seeds, k=int(k_hop))

    # Personalized PageRank on the k-hop subgraph for efficiency
    sub = G.subgraph(k2_neighborhood).copy()
    pers = {}
    total = 0.0
    for n in valid_seeds:
        if n in sub:
            pers[n] = 1.0
            total += 1.0
    if total == 0:
        # fallback uniform personalization across sub nodes
        for n in sub.nodes():
            pers[n] = 1.0
        total = len(pers)
    for n in list(pers.keys()):
        pers[n] = pers[n] / total if total > 0 else 0.0

    try:
        ppr_scores = nx.pagerank(sub, alpha=0.85, personalization=pers, weight="total_amount", max_iter=100, tol=1e-06)
    except Exception:
        # as a last resort, compute PageRank on the full graph but masked by neighborhood
        try:
            ppr_scores = nx.pagerank(G, alpha=0.85, personalization=pers, weight="total_amount", max_iter=100, tol=1e-06)
        except Exception:
            # If pagerank still fails (e.g., missing scipy), use a simple fallback distribution
            ppr_scores = {n: 0.0 for n in sub.nodes()}
            for s in valid_seeds:
                if s in ppr_scores:
                    ppr_scores[s] = 1.0 / len(valid_seeds)

    # Investigation score (simple version)
    results = []
    # precompute max degree for normalization
    try:
        max_deg = max(dict(G.degree(weight='total_amount')).values())
    except Exception:
        max_deg = 1
    for n in k2_neighborhood:
        deg = G.degree(n, weight='total_amount')
        ppr = ppr_scores.get(n, 0.0)
        is_seed = n in valid_seeds
        score = 0.4 * (1.0 if is_seed else 0.0) + 0.3 * ppr + 0.3 * (deg / max(1, max_deg))
        results.append({
            'account_id': n,
            'investigation_score': float(score),
            'ppr_score': float(ppr),
            'degree': int(deg),
            'is_seed': bool(is_seed)
        })
    results = sorted(results, key=lambda x: x['investigation_score'], reverse=True)[:200]

    # Build node and edge lists for the subgraph to return to frontend
    nodes_out = []
    for n in k2_neighborhood:
        nodes_out.append({
            'account_id': n,
            'weighted_degree': G.nodes[n].get('weighted_degree', 0),
            'inflow': G.nodes[n].get('inflow', 0),
            'outflow': G.nodes[n].get('outflow', 0),
            'is_seed': n in valid_seeds
        })

    edges_out = []
    for u, v, d in G.edges(data=True):
        if u in k2_neighborhood and v in k2_neighborhood:
            edges_out.append({
                'from': u,
                'to': v,
                'tx_count': int(d.get('tx_count', 0)),
                'total_amount': float(d.get('total_amount', 0.0)),
                'last_tx': str(d.get('last_tx', ''))
            })

    return {
        'suspects': results,
        'nodes': nodes_out,
        'edges': edges_out,
        'invalid_seeds': invalid_seeds,
        'graph_stats': {'nodes': G.number_of_nodes(), 'edges': G.number_of_edges()}
    }

@app.get("/")
def root():
    return {"message": "AML FastAPI backend running."}
