# SynergiX_Team404
# Money Laundering Pattern Detection
Overview
--------

Two-phase prototype for detecting and investigating money-laundering activity in transaction data using graph-based methods.

- Phase 1 (GAT Model): A Graph Attention Network (GAT) based GUI and training notebook for detecting suspicious transactions.
- Phase 2 (Investigation): Graph analysis and investigation pipeline + a FastAPI backend to explore suspects and neighborhoods.

Repository structure
--------------------

- `Phase1 (GAT Model)/`
	- `money_laundering_detector.py` — Tkinter GUI that loads a trained GAT model, preprocesses transactions, builds a graph, and produces predictions and visualizations.
	- `SynrgiXAML_Model_Training.ipynb` — Jupyter notebook containing model training, evaluation, and diagnostics for the GAT model.

- `Phase2_Investigation/`
	- `aml_pipeline.py` — Command-line pipeline that builds a transaction graph, computes personalized PageRank, structural features, and exports investigation outputs to `aml_output/`.
	- `backend.py` — FastAPI backend exposing an `/investigate/` endpoint and serving static frontend assets. It pre-builds a graph from `HI-Small_Trans.csv` and caches it as `graph.gpickle`.

Getting started
---------------

Prerequisites
- Python 3.8 or newer
- pip and virtualenv (recommended)

Create a virtual environment and install common packages (example):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas numpy networkx matplotlib seaborn scikit-learn tqdm jupyter
# Optional / required for Phase1 GUI model: torch torchvision torchaudio torch-geometric
# Optional / required for Phase2 backend: fastapi uvicorn
```

If this project provides a `requirements.txt` later, prefer `pip install -r requirements.txt`.

Quick usage
-----------

1) Phase 1 — GUI detector

 - Open the training notebook to inspect data and model config:

```powershell
jupyter notebook "Phase1 (GAT Model)\SynrgiXAML_Model_Training.ipynb"
```

 - Run the detector GUI (ensure model and preprocess artifacts paths inside `money_laundering_detector.py` are valid):

```powershell
python "Phase1 (GAT Model)\money_laundering_detector.py"
```

Notes:
 - `money_laundering_detector.py` expects trained model checkpoint and preprocessing pickles (scalers, label encoders, account mappings). Update the hard-coded paths near the top of the file or place the files where the script expects them.
 - The GUI provides data upload, processing, predictions, and visualization tabs.

2) Phase 2 — Investigation pipeline

 - The pipeline reads a CSV named `HI-Small_Trans.csv` by default. To run it:

```powershell
python "Phase2_Investigation\aml_pipeline.py"
```

 - Outputs are saved to the `aml_output/` directory by default (seed accounts, top suspects, Neo4j import CSVs, evidence JSON).

3) Phase 2 — FastAPI backend

 - Start the backend (from the repository root or `Phase2_Investigation` folder):

```powershell
cd "Phase2_Investigation"; uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

 - The server builds (or loads) a cached graph file `graph.gpickle`. The `/investigate/` POST endpoint accepts a seed file or a comma-separated list of suspicious accounts and a `k_hop` form value and returns suspects, nodes, and edges in JSON.

Tips & notes
------------

- Inspect and adapt file paths in the scripts before running, especially model/checkpoint and CSV locations.
- The training notebook documents dataset assumptions and feature engineering used by the GUI and pipeline. Re-run preprocessing steps there if you retrain the model.
- If you plan to deploy the FastAPI backend, add a small `requirements.txt` and consider running behind a production server.

Where outputs go
----------------

- `Phase2_Investigation/aml_output/` — pipeline CSVs and JSON evidence files
- `Phase2_Investigation/graph.gpickle` — cached graph used by the backend

Contributing
------------

If you'd like help improving this README, adding a `requirements.txt`, or making the startup experience easier (e.g., relative model paths, command-line args), tell me which you prefer and I can implement them.

License
-------

This repository does not include a license file. Add a `LICENSE` if you intend to publish with a specific license.

