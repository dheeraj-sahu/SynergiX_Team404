import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt

CSV_PATH = "HI-Small_Trans.csv"  
TIME_COL = "Timestamp"
FROM_ACC_COL = "Account"
TO_ACC_COL = "Account.1"
AMOUNT_COL = "Amount Received"
FLAG_COL = "Is Laundering"
DATE_FORMAT = None
OUTPUT_DIR = "aml_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_transactions(path=CSV_PATH):
    print("Loading CSV:", path)
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format=DATE_FORMAT, errors="coerce")
    df = df.dropna(subset=[FROM_ACC_COL, TO_ACC_COL, AMOUNT_COL])
    df[AMOUNT_COL] = pd.to_numeric(df[AMOUNT_COL], errors='coerce').fillna(0.0)
    df[FROM_ACC_COL] = df[FROM_ACC_COL].astype(str)
    df[TO_ACC_COL] = df[TO_ACC_COL].astype(str)
    return df

tx = load_transactions()
print("Rows:", len(tx))
tx.head()

def build_graph(df, from_col=FROM_ACC_COL, to_col=TO_ACC_COL, amount_col=AMOUNT_COL, time_col=TIME_COL):
    G = nx.DiGraph()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
        u = row[from_col]
        v = row[to_col]
        amt = float(row[amount_col])
        ts = row[time_col]
        txid = row.get("transaction_id", None) or row.name
        if G.has_edge(u, v):
            G[u][v]["tx_ids"].append(txid)
            G[u][v]["tx_count"] += 1
            G[u][v]["total_amount"] += amt
            if ts is not pd.NaT:
                if "last_tx" not in G[u][v] or pd.isna(G[u][v]["last_tx"]) or ts > G[u][v]["last_tx"]:
                    G[u][v]["last_tx"] = ts
        else:
            G.add_edge(u, v,
                       tx_ids=[txid],
                       tx_count=1,
                       total_amount=amt,
                       first_tx=ts,
                       last_tx=ts)
    for n in G.nodes():
        G.nodes[n]["inflow"] = sum(G[p][n]["total_amount"] for p in G.predecessors(n) if G.has_edge(p,n))
        G.nodes[n]["outflow"] = sum(G[n][q]["total_amount"] for q in G.successors(n) if G.has_edge(n,q))
        G.nodes[n]["weighted_degree"] = G.nodes[n]["inflow"] + G.nodes[n]["outflow"]
    return G

G = build_graph(tx)
print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

def get_seed_scores(df, flag_col=FLAG_COL, from_col=FROM_ACC_COL, to_col=TO_ACC_COL):
    seed_scores = defaultdict(float)
    if flag_col in df.columns:
        for _, r in df.iterrows():
            flag_val = r.get(flag_col, 0)
            try:
                flag_val = float(flag_val)
            except Exception:
                flag_val = 0.0
            if flag_val > 0:
                seed_scores[r[from_col]] = max(seed_scores[r[from_col]], flag_val)
                seed_scores[r[to_col]] = max(seed_scores[r[to_col]], flag_val)
    return dict(seed_scores)

seed_scores = get_seed_scores(tx)
print("Seed accounts from model flags:", len(seed_scores))
pd.DataFrame.from_dict(seed_scores, orient='index', columns=['seed_score'])\
  .reset_index().rename(columns={'index':'account_id'})\
  .to_csv(os.path.join(OUTPUT_DIR, "seed_accounts.csv"), index=False)

def k_hop_neighbors(G, seeds, k=2):
    R = set(seeds)
    frontier = set(seeds)
    for _ in range(k):
        nxt = set()
        for u in frontier:
            nxt |= set(G.predecessors(u))
            nxt |= set(G.successors(u))
        nxt -= R
        R |= nxt
        frontier = nxt
        if not frontier:
            break
    return R

seeds = list(seed_scores.keys())
k2_neighborhood = k_hop_neighbors(G, seeds, k=2)
print("k=2 neighborhood size:", len(k2_neighborhood))

def personalized_pagerank(G, seed_scores, alpha=0.85, weight_attr="total_amount", max_iter=100, tol=1e-06):
    pers = {}
    total = sum(seed_scores.values()) if seed_scores else 0.0
    if total == 0:
        for n in seed_scores.keys():
            pers[n] = 1.0
        total = len(pers)
    for n, v in seed_scores.items():
        pers[n] = v/total if total>0 else 0.0
    # For nodes not in pers, networkx expects 0 values; we pass personalization dict as-is.
    try:
        # networkx may use scipy backend for fast pagerank on large graphs
        ppr = nx.pagerank(G, alpha=alpha, personalization=pers, weight=weight_attr, max_iter=max_iter, tol=tol)
        return ppr
    except ModuleNotFoundError as e:
        # scipy not installed; fallback to power-iteration implementation
        print("scipy not available, using fallback pagerank (power iteration)")
    except Exception as e:
        print("pagerank failed, falling back to power-iteration:", e)

    # Fallback power iteration pagerank (sparse-friendly)
    nodes = list(G.nodes())
    N = len(nodes)
    idx = {n:i for i,n in enumerate(nodes)}
    # build out-link dict with weights
    out_links = {i: {} for i in range(N)}
    for u,v,data in G.edges(data=True):
        ui = idx[u]
        vi = idx[v]
        w = data.get(weight_attr, 1.0)
        out_links[ui][vi] = out_links[ui].get(vi,0.0) + float(w)

    # normalize outlink weights to probabilities
    M = [None]*N
    for i in range(N):
        row = out_links[i]
        total_w = sum(row.values())
        if total_w>0:
            M[i] = {j: val/total_w for j,val in row.items()}
        else:
            M[i] = {}

    # personalization vector
    pr = [0.0]*N
    for n,v in pers.items():
        if n in idx:
            pr[idx[n]] = v
    # start vector: personalization or uniform
    x = pr[:] if sum(pr)>0 else [1.0/N]*N

    for iteration in range(max_iter):
        x_new = [ (1-alpha)*(pr[i] if sum(pr)>0 else 1.0/N) for i in range(N) ]
        # contribution from in-links: we traverse out-links to distribute mass
        for i in range(N):
            row = M[i]
            if not row: continue
            val = x[i]
            for j,w in row.items():
                x_new[j] += alpha * val * w
        # dangling nodes (no out-links) distribute uniformly
        dangling_sum = sum(x[i] for i in range(N) if not M[i])
        if dangling_sum:
            add = alpha * dangling_sum / N
            for j in range(N):
                x_new[j] += add

        # check convergence
        err = sum(abs(x_new[i]-x[i]) for i in range(N))
        x = x_new
        if err < tol:
            break

    ppr = {nodes[i]: x[i] for i in range(N)}
    return ppr

ppr_scores = personalized_pagerank(G, seed_scores)
top_ppr = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)[:15]
print("Top PPR nodes:", top_ppr)

def compute_structural_features(G, sample_betw_nodes=200):
    deg = dict(G.degree(weight='total_amount'))
    max_deg = max(deg.values()) if deg else 1.0
    deg_norm = {n: deg.get(n,0.0)/max_deg for n in G.nodes()}
    try:
        btw = nx.betweenness_centrality(G, k=min(sample_betw_nodes, G.number_of_nodes()), normalized=True, seed=42)
    except Exception as e:
        print("Betweenness failed (large graph); setting zeros.", e)
        btw = {n:0.0 for n in G.nodes()}
    try:
        core = nx.core_number(G.to_undirected())
        max_core = max(core.values()) if core else 1
        core_norm = {n: core.get(n,0)/max_core for n in G.nodes()}
    except Exception:
        core_norm = {n:0.0 for n in G.nodes()}
    return deg_norm, btw, core_norm

deg_norm, btw, core_norm = compute_structural_features(G)

def find_short_cycles(G, seeds, max_len=6, neighborhood_k=2, max_cycles_per_seed=50):
    cycles_set = []
    for s in tqdm(seeds, desc="Cycle detection per seed"):
        if s not in G:
            continue
        nbrs = k_hop_neighbors(G, [s], k=neighborhood_k)
        sub = G.subgraph(nbrs).copy()
        try:
            cnt = 0
            for cyc in nx.simple_cycles(sub):
                if len(cyc) <= max_len:
                    cycles_set.append(cyc)
                    cnt += 1
                if cnt >= max_cycles_per_seed:
                    break
        except Exception as e:
            continue
    return cycles_set

cycles = find_short_cycles(G, seeds, max_len=6, neighborhood_k=2, max_cycles_per_seed=20)
print("Found cycles (count):", len(cycles))

def build_investigation_scores(G, ppr, deg_norm, btw, core_norm, seed_scores, cycles):
    node_cycle_presence = {n:0 for n in G.nodes()}
    for cyc in cycles:
        for n in cyc:
            node_cycle_presence[n] = 1
    max_btw = max(btw.values()) if btw else 1.0
    scores = {}
    for n in G.nodes():
        s_model = seed_scores.get(n, 0.0)
        s_ppr = ppr.get(n, 0.0)
        s_deg = deg_norm.get(n, 0.0)
        s_btw = btw.get(n, 0.0) / (max_btw if max_btw else 1.0)
        s_core = core_norm.get(n, 0.0)
        s_cycle = node_cycle_presence.get(n, 0.0)
        score = 0.4*s_model + 0.30*s_ppr + 0.10*s_deg + 0.10*s_btw + 0.05*s_core + 0.05*s_cycle
        scores[n] = float(score)
    return scores

invest_scores = build_investigation_scores(G, ppr_scores, deg_norm, btw, core_norm, seed_scores, cycles)
topK = sorted(invest_scores.items(), key=lambda x: x[1], reverse=True)[:200]
top_df = pd.DataFrame(topK, columns=['account_id','investigation_score'])
top_df = top_df.merge(pd.DataFrame.from_dict(seed_scores, orient='index', columns=['seed_score']).reset_index().rename(columns={'index':'account_id'}), on='account_id', how='left')
top_df.to_csv(os.path.join(OUTPUT_DIR, "top_suspects.csv"), index=False)
print("Top suspects saved to:", os.path.join(OUTPUT_DIR, "top_suspects.csv"))
top_df.head(30)

def evidence_trail_for_account(G, account_id, depth=3, max_paths=50):
    trails = []
    if account_id not in G:
        return trails
    for s in seeds:
        if s not in G: continue
        try:
            for p in nx.all_simple_paths(G, source=s, target=account_id, cutoff=depth):
                trails.append({"type":"seed_to_account", "seed":s, "path":p})
                if len(trails)>=max_paths: break
        except Exception:
            continue
        if len(trails)>=max_paths: break
    out = []
    for nbr in G.successors(account_id):
        e = G[account_id][nbr]
        out.append({"to":nbr, "tx_count":e.get("tx_count",0), "total_amount":e.get("total_amount",0)})
    return {"paths_from_seeds":trails, "outgoing_summary":sorted(out, key=lambda x: x["total_amount"], reverse=True)[:10]}

evidence = {}
for acct in top_df.head(10)['account_id'].tolist():
    evidence[acct] = evidence_trail_for_account(G, acct, depth=4, max_paths=30)
with open(os.path.join(OUTPUT_DIR, "evidence_top10.json"), "w") as f:
    json.dump(evidence, f, default=str, indent=2)
print("Evidence (top10) saved to:", os.path.join(OUTPUT_DIR, "evidence_top10.json"))

nodes = set()
for u, v, data in G.edges(data=True):
    nodes.add(u); nodes.add(v)
node_rows = [{"account_id":n, "weighted_degree": G.nodes[n].get("weighted_degree", 0)} for n in nodes]
pd.DataFrame(node_rows).to_csv(os.path.join(OUTPUT_DIR,"neo4j_nodes.csv"), index=False)

edge_rows = []
for u,v,d in G.edges(data=True):
    edge_rows.append({
        "from_account_id": u,
        "to_account_id": v,
        "tx_count": d.get("tx_count",0),
        "total_amount": d.get("total_amount",0),
        "last_tx": d.get("last_tx", "")
    })
pd.DataFrame(edge_rows).to_csv(os.path.join(OUTPUT_DIR, "neo4j_edges.csv"), index=False)
print("Neo4j CSVs exported to:", OUTPUT_DIR)

print("Pipeline complete. Outputs in:", OUTPUT_DIR)
