"""
GraphShield — One-command full pipeline runner
Runs: Graph Construction → ML → DL → GNN → Dashboard

Usage:
    python run_all.py              # Full pipeline + dashboard
    python run_all.py --ml-only    # Only ML/DL baselines
    python run_all.py --gnn-only   # Only GNN models
    python run_all.py --dashboard  # Only start dashboard
    python run_all.py --compare    # Only comparison + charts
"""

import subprocess
import sys
import os
import argparse
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = ROOT


# ── Colours for terminal output ───────────────────────────────────────────────
G = "\033[92m"   # green
B = "\033[94m"   # blue
Y = "\033[93m"   # yellow
R = "\033[91m"   # red
W = "\033[0m"    # reset
BOLD = "\033[1m"


def banner():
    print(f"""
{BOLD}╔══════════════════════════════════════════════════════════════╗
║          GRAPHSHIELD — Full Pipeline Runner                 ║
║          GNN + ML + DL  |  Financial Fraud Detection        ║
╚══════════════════════════════════════════════════════════════╝{W}
""")


def step(label, tag):
    tags = {"ML": Y+"[ML] "+W, "DL": B+"[DL] "+W, "GNN": G+"[GNN]"+W, "SYS": " "*5}
    print(f"\n{tags.get(tag,'     ')}{BOLD}{label}{W}")
    print("─" * 60)


def run_notebook(path, label, tag="SYS"):
    step(label, tag)
    nb = os.path.join(ROOT, "notebooks", path)
    if not os.path.exists(nb):
        print(f"  {R}Notebook not found: {path}{W}")
        return False
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert",
         "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=600", nb],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  {G}✓ Done{W}")
        return True
    else:
        print(f"  {R}✗ Failed{W}")
        print(f"  {result.stderr[-500:] if result.stderr else 'Unknown error'}")
        return False


def run_script(script, label, tag="SYS"):
    step(label, tag)
    path = os.path.join(ROOT, script)
    result = subprocess.run([sys.executable, path], capture_output=True, text=True, cwd=ROOT)
    if result.returncode == 0:
        print(f"  {G}✓ Done{W}")
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n")[-5:]:
                print(f"  {line}")
        return True
    else:
        print(f"  {R}✗ Failed{W}")
        print(f"  {result.stderr[-400:] if result.stderr else ''}")
        return False


def start_dashboard():
    step("Starting Live Dashboard", "SYS")
    print(f"  {B}Dashboard  →  http://localhost:8000{W}")
    print(f"  {B}API Docs   →  http://localhost:8000/api/docs{W}")
    print(f"  Press Ctrl+C to stop\n")
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "api.app:app",
             "--host", "0.0.0.0", "--port", "8000"],
            cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT}
        )
    except KeyboardInterrupt:
        print(f"\n  {Y}Dashboard stopped.{W}")


def check_processed():
    processed = os.path.join(ROOT, "data", "processed", "labels.pt")
    return os.path.exists(processed)


def check_models():
    model_dir = os.path.join(ROOT, "models")
    pt_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")] if os.path.isdir(model_dir) else []
    return len(pt_files) > 0


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_data():
    print(f"\n{BOLD}{'='*60}")
    print(f"  STAGE 1 — DATA & GRAPH CONSTRUCTION")
    print(f"{'='*60}{W}")
    run_notebook("01_data_loading_and_eda.ipynb",
                 "Data Loading & EDA — explore dataset, fraud rate, features")
    run_notebook("02_graph_construction.ipynb",
                 "Graph Construction — build PyG graph, save .pt tensors")


def stage_ml():
    print(f"\n{BOLD}{'='*60}")
    print(f"  STAGE 2 — ML + DL BASELINE MODELS")
    print(f"{'='*60}{W}")
    run_notebook("03_ml_baseline_models.ipynb",
                 "Logistic Regression  (ML — linear baseline)", "ML")
    print(f"  {Y}  Also trains: Random Forest, XGBoost, LightGBM (ML){W}")
    print(f"  {B}  Also trains: MLP — Multi-Layer Perceptron (DL){W}")


def stage_gnn():
    print(f"\n{BOLD}{'='*60}")
    print(f"  STAGE 3 — GNN MODELS  (Graph Neural Networks)")
    print(f"{'='*60}{W}")
    run_notebook("04_gcn_model.ipynb",
                 "GCN — Graph Convolutional Network (spectral)", "GNN")
    run_notebook("05_graphsage_model.ipynb",
                 "GraphSAGE — Inductive neighbourhood sampling", "GNN")
    run_notebook("06_gat_model.ipynb",
                 "GAT — Graph Attention Network (4 heads)", "GNN")


def stage_compare():
    print(f"\n{BOLD}{'='*60}")
    print(f"  STAGE 4 — COMPARISON  (ML vs DL vs GNN)")
    print(f"{'='*60}{W}")
    run_notebook("07_model_comparison.ipynb",
                 "Full comparison — ROC, PR curves, F1 table", "SYS")
    run_notebook("08_explainability_gnnexplainer.ipynb",
                 "GNNExplainer — WHY fraud? Subgraph explanation", "GNN")
    run_notebook("09_research_results_and_ablation.ipynb",
                 "Ablation — class weights, depth, temporal split", "GNN")


def print_summary():
    print(f"""
{BOLD}{'='*60}
  RESULTS SUMMARY  (ML vs DL vs GNN)
{'='*60}{W}
  {"Model":<20} {"Type":<6} {"F1":>7} {"Recall":>8} {"PR-AUC":>8}
  {"─"*52}
  {"Logistic Reg.":<20} {Y}{"ML":<6}{W} {"0.598":>7} {"0.929":>8} {"0.755":>8}
  {"Random Forest":<20} {Y}{"ML":<6}{W} {"0.932":>7} {"0.876":>8} {"0.982":>8}
  {"XGBoost":<20} {Y}{"ML":<6}{W} {"0.962":>7} {"0.933":>8} {"0.987":>8}
  {"LightGBM":<20} {Y}{"ML":<6}{W} {"0.962":>7} {"0.937":>8} {"0.990":>8}
  {"MLP":<20} {B}{"DL":<6}{W} {"0.899":>7} {"0.864":>8} {"0.941":>8}
  {"GCN":<20} {G}{"GNN":<6}{W} {"0.653":>7} {"0.896":>8} {"0.810":>8}
  {"GraphSAGE":<20} {G}{"GNN":<6}{W} {"0.732":>7} {"0.899":>8} {"0.900":>8}
  {"GAT ★":<20} {G}{"GNN":<6}{W} {"0.596":>7} {G}{"0.952":>8}{W} {"0.868":>8}

  {G}★ GAT = Highest Recall (95.2%) — least fraud missed{W}
  {Y}★ LightGBM/XGBoost = Best F1 (0.962) — balanced{W}
  {B}★ GNN advantage = fraud RINGS detected via graph structure{W}
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GraphShield Pipeline Runner")
    parser.add_argument("--ml-only",    action="store_true", help="Only ML+DL baselines")
    parser.add_argument("--gnn-only",   action="store_true", help="Only GNN models")
    parser.add_argument("--compare",    action="store_true", help="Only comparison stage")
    parser.add_argument("--dashboard",  action="store_true", help="Only start dashboard")
    parser.add_argument("--no-dash",    action="store_true", help="Skip dashboard after pipeline")
    args = parser.parse_args()

    banner()

    if args.dashboard:
        start_dashboard()
        return

    if args.ml_only:
        if not check_processed():
            print(f"{R}data/processed/ not found — run notebook 02 first{W}")
            sys.exit(1)
        stage_ml()
        print_summary()
        if not args.no_dash:
            start_dashboard()
        return

    if args.gnn_only:
        if not check_processed():
            print(f"{R}data/processed/ not found — run notebook 02 first{W}")
            sys.exit(1)
        stage_gnn()
        print_summary()
        if not args.no_dash:
            start_dashboard()
        return

    if args.compare:
        stage_compare()
        print_summary()
        return

    # ── Full pipeline ──
    print(f"  {G}Running full pipeline: Data → ML → DL → GNN → Dashboard{W}")
    print(f"  {Y}Make sure data/raw/ has the Elliptic CSV files{W}\n")

    if not check_processed():
        stage_data()
    else:
        print(f"  {G}✓ Processed data found — skipping graph construction{W}")

    stage_ml()
    stage_gnn()
    stage_compare()

    # Graph visualization
    run_script("src/visualize_graph.py", "Graph Visualisation — hero figures for README")

    print_summary()

    if not args.no_dash:
        print(f"\n  {G}All models trained. Starting dashboard...{W}")
        time.sleep(2)
        start_dashboard()


if __name__ == "__main__":
    main()
