"""
report.py
Generates a self-contained HTML training report.

Sections
────────
1. Executive Summary
2. Dataset Description
3. Model Architecture
4. Training Method & Hyperparameters
5. Training Results (loss tables + embedded plots)
6. Evaluation Metrics

Usage
─────
    python report.py
    # → reports/training_report.html
"""

import base64
import json
from datetime import datetime
from pathlib import Path

BASE      = Path(__file__).parent
OUT_DIR   = BASE / "outputs"
PLOT_DIR  = OUT_DIR / "plots"
REPORT_DIR = BASE / "reports"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def img_to_b64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def embed_img(path: Path, caption: str = "", width: str = "100%") -> str:
    b64 = img_to_b64(path)
    if not b64:
        return f'<p class="missing">[Plot not yet available: {path.name}]</p>'
    return (
        f'<figure style="margin:0 0 1.5rem 0">'
        f'<img src="data:image/png;base64,{b64}" style="width:{width};border-radius:6px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.12)" alt="{caption}">'
        f'{"<figcaption>" + caption + "</figcaption>" if caption else ""}'
        f'</figure>'
    )


def table_from_dict(d: dict, header1="Parameter", header2="Value") -> str:
    rows = "".join(
        f"<tr><td>{k}</td><td><code>{v}</code></td></tr>"
        for k, v in d.items()
    )
    return (
        f'<table><thead><tr><th>{header1}</th><th>{header2}</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def loss_table(history: dict) -> str:
    epochs = history["epoch"]
    best_val_idx = int(min(range(len(epochs)),
                           key=lambda i: history["val_total"][i]))
    rows = ""
    for i, e in enumerate(epochs):
        best = " class='best-row'" if i == best_val_idx else ""
        rows += (
            f"<tr{best}>"
            f"<td>{e}</td>"
            f"<td>{history['lr'][i]:.2e}</td>"
            f"<td>{history['train_total'][i]:.5f}</td>"
            f"<td>{history['train_l1'][i]:.5f}</td>"
            f"<td>{history['train_kl'][i]:.5f}</td>"
            f"<td>{history['val_total'][i]:.5f}</td>"
            f"<td>{history['val_l1'][i]:.5f}</td>"
            f"<td>{history['val_kl'][i]:.5f}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Epoch</th><th>LR</th>"
        "<th>Train Total</th><th>Train L1</th><th>Train KL</th>"
        "<th>Val Total</th><th>Val L1</th><th>Val KL</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )


# ─── Report builder ───────────────────────────────────────────────────────────

def build_report():
    # Load data
    cfg_path = OUT_DIR / "run_config.json"
    met_path = OUT_DIR / "metrics.json"

    cfg     = json.load(open(cfg_path)) if cfg_path.exists() else {}
    history = json.load(open(met_path)) if met_path.exists() else {}
    meta    = cfg.get("dataset", {})
    model   = cfg.get("model", {})
    train   = cfg.get("training", {})

    # Best / final losses
    if history:
        best_val   = min(history["val_total"])
        best_epoch = history["epoch"][history["val_total"].index(best_val)]
        final_train = history["train_total"][-1]
        final_val   = history["val_total"][-1]
        total_epochs = len(history["epoch"])
    else:
        best_val = best_epoch = final_train = final_val = total_epochs = "N/A"

    # ── CSS ──────────────────────────────────────────────────────────────────
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px; color: #1e293b; background: #f1f5f9; padding: 2rem;
    }
    .page { max-width: 1100px; margin: 0 auto; }
    h1 { font-size: 2rem; color: #0f172a; margin-bottom: 0.25rem; }
    h2 { font-size: 1.2rem; color: #1e40af; border-bottom: 2px solid #bfdbfe;
         padding-bottom: 0.4rem; margin: 2rem 0 1rem; }
    h3 { font-size: 1rem; color: #374151; margin: 1rem 0 0.5rem; }
    .subtitle { color: #64748b; font-size: 0.9rem; margin-bottom: 2rem; }
    .card { background: white; border-radius: 10px; padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 1.5rem; }
    .summary-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
    .kpi { background: white; border-radius: 8px; padding: 1rem 1.25rem;
           box-shadow: 0 1px 3px rgba(0,0,0,.08); text-align: center; }
    .kpi-val { font-size: 1.6rem; font-weight: 700; color: #1e40af; }
    .kpi-lbl { font-size: 0.75rem; color: #64748b; margin-top: 0.2rem; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 0.5rem; }
    th { background: #1e40af; color: white; padding: 6px 10px; text-align: left; }
    td { padding: 5px 10px; border-bottom: 1px solid #e2e8f0; }
    tr:hover td { background: #f0f9ff; }
    .best-row td { background: #dcfce7 !important; font-weight: 600; }
    code { background: #f1f5f9; padding: 1px 4px; border-radius: 3px;
           font-family: 'JetBrains Mono', monospace; font-size: 12px; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    figcaption { font-size: 0.8rem; color: #64748b; text-align: center; margin-top: 4px; }
    .missing { color: #94a3b8; font-style: italic; padding: 1rem;
               background: #f8fafc; border-radius: 6px; text-align: center; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 999px;
           font-size: 11px; font-weight: 600; }
    .tag-blue { background:#dbeafe; color:#1e40af; }
    .tag-green { background:#dcfce7; color:#15803d; }
    .tag-purple { background:#f3e8ff; color:#7e22ce; }
    .arch-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
                padding: 1rem; font-family: monospace; font-size: 12px; line-height: 1.8; }
    """

    # ── KPI cards ────────────────────────────────────────────────────────────
    kpis = f"""
    <div class="summary-grid">
        <div class="kpi"><div class="kpi-val">{meta.get('n_train','N/A')+meta.get('n_val',0) if isinstance(meta.get('n_train'),int) else 'N/A'}</div><div class="kpi-lbl">Total Episodes</div></div>
        <div class="kpi"><div class="kpi-val">{model.get('n_params','N/A'):,}</div><div class="kpi-lbl">Model Parameters</div></div>
        <div class="kpi"><div class="kpi-val">{total_epochs}</div><div class="kpi-lbl">Epochs Trained</div></div>
        <div class="kpi"><div class="kpi-val">{f'{best_val:.4f}' if isinstance(best_val,float) else best_val}</div><div class="kpi-lbl">Best Val Loss</div></div>
    </div>
    """

    # ── Dataset section ───────────────────────────────────────────────────────
    dataset_params = {
        "Total episodes":        f"{meta.get('n_train','N/A')} train  /  {meta.get('n_val','N/A')} val",
        "Timesteps per episode": meta.get("episode_len", "N/A"),
        "Observation dim":       meta.get("obs_dim", "N/A"),
        "Action dim":            meta.get("action_dim", "N/A"),
        "Image size":            f"{meta.get('img_size','N/A')} × {meta.get('img_size','N/A')} px  (RGB)",
        "Action chunk size":     meta.get("chunk_size", "N/A"),
        "Noise scale":           meta.get("noise_scale", "N/A"),
        "Task phases":           "  →  ".join(meta.get("phases", [])),
        "Format":                "NumPy .npz per episode",
    }

    # ── Model section ─────────────────────────────────────────────────────────
    model_params = {
        "Architecture":      "ACT (Action Chunking with Transformers)",
        "Hidden dim":        model.get("hidden_dim", "N/A"),
        "Latent dim (CVAE)": model.get("latent_dim", "N/A"),
        "Attention heads":   model.get("nhead", "N/A"),
        "Encoder layers":    model.get("num_enc_layers", "N/A"),
        "Decoder layers":    model.get("num_dec_layers", "N/A"),
        "Chunk size":        model.get("chunk_size", "N/A"),
        "Total parameters":  f"{model.get('n_params','N/A'):,}",
        "Image encoder":     "3-layer CNN → Linear projection",
        "State encoder":     "Linear → hidden_dim",
    }

    arch_diagram = """
<b>Training path:</b>
  obs (state + image) ──────────────────────────────────────────┐
                                                                 ├──► Transformer encoder ──► CVAE decoder ──► action chunk
  action_seq ──► CVAE encoder ──► z ~ N(mu, sigma) ────────────┘

<b>Inference path:</b>
  obs ──► Transformer encoder ──► CVAE decoder (z = 0) ──► action chunk

<b>Loss:</b>  L = L1(pred, gt) + β · KL( N(μ,σ) ‖ N(0,I) )
    """

    # ── Training section ──────────────────────────────────────────────────────
    train_params = {
        "Optimiser":          "AdamW",
        "Learning rate":      train.get("lr", "N/A"),
        "Weight decay":       train.get("weight_decay", "N/A"),
        "LR schedule":        "Linear warm-up + cosine annealing",
        "Warm-up epochs":     train.get("warmup_epochs", "N/A"),
        "Epochs":             train.get("epochs", "N/A"),
        "Batch size":         train.get("batch_size", "N/A"),
        "Gradient clipping":  train.get("grad_clip", "N/A"),
        "KL weight (β)":      train.get("beta", "N/A"),
        "Checkpoint saved every": f"{train.get('save_every','N/A')} epochs",
    }

    # ── Results section ───────────────────────────────────────────────────────
    result_params = {
        "Best epoch":       best_epoch,
        "Best val loss":    f"{best_val:.5f}" if isinstance(best_val, float) else best_val,
        "Final train loss": f"{final_train:.5f}" if isinstance(final_train, float) else final_train,
        "Final val loss":   f"{final_val:.5f}"  if isinstance(final_val, float)  else final_val,
    }

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ACT Policy Training Report</title>
<style>{css}</style>
</head>
<body>
<div class="page">

  <h1>ACT Policy Training Report</h1>
  <p class="subtitle">
    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
    <span class="tag tag-blue">PyTorch {_try_torch_version()}</span> &nbsp;
    <span class="tag tag-green">ACT</span> &nbsp;
    <span class="tag tag-purple">Cable Insertion</span>
  </p>

  <!-- ── 1. Executive Summary ─────────────────────────────────────────────── -->
  <h2>1. Executive Summary</h2>
  {kpis}
  <div class="card">
    {table_from_dict(result_params, "Metric", "Value")}
  </div>

  <!-- ── 2. Dataset ────────────────────────────────────────────────────────── -->
  <h2>2. Dataset</h2>
  <div class="card">
    <p style="margin-bottom:1rem;color:#475569">
      Synthetic dataset generated by simulating a 6-DOF UR-style robot arm
      performing a cable-insertion task. Each episode follows a 6-phase
      trajectory (home → pre-grasp → grasp → lift → approach → insert) with
      per-episode random offsets and Gaussian timestep noise for variation.
    </p>
    {table_from_dict(dataset_params)}
  </div>
  <div class="card">
    {embed_img(PLOT_DIR / '04_dataset_overview.png', 'Dataset overview: joint trajectories, TCP pose, action distribution, and sample frames')}
  </div>

  <!-- ── 3. Model Architecture ─────────────────────────────────────────────── -->
  <h2>3. Model Architecture</h2>
  <div class="two-col">
    <div class="card">{table_from_dict(model_params)}</div>
    <div class="card">
      <h3>Architecture diagram</h3>
      <pre class="arch-box">{arch_diagram}</pre>
    </div>
  </div>

  <!-- ── 4. Training Method ─────────────────────────────────────────────────── -->
  <h2>4. Training Method &amp; Hyperparameters</h2>
  <div class="card">
    <p style="margin-bottom:1rem;color:#475569">
      Training follows the original ACT paper (Zhao et al., 2023).
      A CVAE encoder compresses the ground-truth action chunk into a latent
      vector <em>z</em>; the decoder then predicts the chunk from the current
      observation plus <em>z</em>. At inference time <em>z = 0</em> is used.
      Loss = L1 reconstruction + β·KL divergence.
    </p>
    {table_from_dict(train_params)}
  </div>

  <!-- ── 5. Training Results ────────────────────────────────────────────────── -->
  <h2>5. Training Results</h2>
  <div class="card">
    {embed_img(PLOT_DIR / '01_loss_curves.png',      'Total training vs validation loss over epochs')}
  </div>
  <div class="two-col">
    <div class="card">
      {embed_img(PLOT_DIR / '02_loss_components.png', 'L1 and KL loss components')}
    </div>
    <div class="card">
      {embed_img(PLOT_DIR / '03_lr_curve.png',        'Learning-rate schedule')}
    </div>
  </div>

  <!-- ── 6. Evaluation ─────────────────────────────────────────────────────── -->
  <h2>6. Evaluation — Action Predictions</h2>
  <div class="card">
    {embed_img(PLOT_DIR / '05_action_predictions.png', 'Model action chunk predictions (hatched) vs ground truth (solid) for 4 timesteps in a held-out episode')}
  </div>

  <!-- ── 7. Full Epoch Log ──────────────────────────────────────────────────── -->
  <h2>7. Full Epoch Loss Log</h2>
  <div class="card" style="overflow-x:auto">
    {loss_table(history) if history else '<p class="missing">Metrics not available yet.</p>'}
  </div>

</div>
</body>
</html>"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "training_report.html"
    out.write_text(html)
    print(f"Report saved to {out}")
    return out


def _try_torch_version():
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    build_report()
