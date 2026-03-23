"""
report_lerobot.py
Generates the HTML training report for the LeRobot ACT run.
"""

import base64, json
from datetime import datetime
from pathlib import Path

BASE      = Path(__file__).parent
OUT_DIR   = BASE / "outputs_lerobot"
PLOT_DIR  = OUT_DIR / "plots"
REPORT_DIR = BASE / "reports"


def b64(path):
    if not path.exists(): return ""
    return base64.b64encode(open(path,"rb").read()).decode()

def embed(path, caption="", width="100%"):
    d = b64(path)
    if not d:
        return f'<p class="missing">[{path.name} not yet generated]</p>'
    return (f'<figure><img src="data:image/png;base64,{d}" '
            f'style="width:{width};border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.12)" '
            f'alt="{caption}">{"<figcaption>"+caption+"</figcaption>" if caption else ""}'
            f'</figure>')

def tbl(d, h1="Parameter", h2="Value"):
    rows = "".join(f"<tr><td>{k}</td><td><code>{v}</code></td></tr>" for k,v in d.items())
    return f'<table><thead><tr><th>{h1}</th><th>{h2}</th></tr></thead><tbody>{rows}</tbody></table>'

def loss_tbl(history):
    if not history: return '<p class="missing">No metrics yet.</p>'
    best_i = min(range(len(history)), key=lambda i: history[i].get("val_loss", history[i].get("val_l1_loss", 9e9)))
    rows = ""
    for i, r in enumerate(history):
        t  = r.get("train_loss",  r.get("train_l1_loss", "-"))
        v  = r.get("val_loss",    r.get("val_l1_loss",   "-"))
        tl = r.get("train_l1_loss", "-")
        tk = r.get("train_kld_loss", "-")
        vl = r.get("val_l1_loss",   "-")
        vk = r.get("val_kld_loss",  "-")
        best = " class='best-row'" if i == best_i else ""
        def fmt(x): return f"{x:.5f}" if isinstance(x, float) else x
        rows += (f"<tr{best}><td>{r['epoch']}</td><td>{r['lr']:.2e}</td>"
                 f"<td>{fmt(t)}</td><td>{fmt(tl)}</td><td>{fmt(tk)}</td>"
                 f"<td>{fmt(v)}</td><td>{fmt(vl)}</td><td>{fmt(vk)}</td></tr>")
    return (f'<table><thead><tr><th>Epoch</th><th>LR</th>'
            f'<th>T-Total</th><th>T-L1</th><th>T-KL</th>'
            f'<th>V-Total</th><th>V-L1</th><th>V-KL</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     font-size:14px;color:#1e293b;background:#f1f5f9;padding:2rem}
.page{max-width:1100px;margin:0 auto}
h1{font-size:2rem;color:#0f172a;margin-bottom:.25rem}
h2{font-size:1.2rem;color:#1e40af;border-bottom:2px solid #bfdbfe;
   padding-bottom:.4rem;margin:2rem 0 1rem}
h3{font-size:1rem;color:#374151;margin:1rem 0 .5rem}
.sub{color:#64748b;font-size:.9rem;margin-bottom:2rem}
.card{background:white;border-radius:10px;padding:1.5rem;
      box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:1.5rem}
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem}
.kpi{background:white;border-radius:8px;padding:1rem 1.25rem;
     box-shadow:0 1px 3px rgba(0,0,0,.08);text-align:center}
.kv{font-size:1.6rem;font-weight:700;color:#1e40af}
.kl{font-size:.75rem;color:#64748b;margin-top:.2rem}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:.5rem}
th{background:#1e40af;color:white;padding:6px 10px;text-align:left}
td{padding:5px 10px;border-bottom:1px solid #e2e8f0}
tr:hover td{background:#f0f9ff}
.best-row td{background:#dcfce7!important;font-weight:600}
code{background:#f1f5f9;padding:1px 4px;border-radius:3px;font-size:12px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
figcaption{font-size:.8rem;color:#64748b;text-align:center;margin-top:4px}
.missing{color:#94a3b8;font-style:italic;padding:1rem;background:#f8fafc;
         border-radius:6px;text-align:center}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600}
.tb{background:#dbeafe;color:#1e40af}
.tg{background:#dcfce7;color:#15803d}
.tp{background:#f3e8ff;color:#7e22ce}
pre{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
    padding:1rem;font-size:12px;line-height:1.8;overflow-x:auto}
"""

def build():
    cfg_path = OUT_DIR / "run_config.json"
    met_path = OUT_DIR / "metrics.json"
    cfg  = json.load(open(cfg_path)) if cfg_path.exists() else {}
    hist = json.load(open(met_path)) if met_path.exists() else []

    m   = cfg.get("model",    {})
    tr  = cfg.get("training", {})
    ds  = cfg.get("dataset",  {})

    best_val  = min((r.get("val_loss", r.get("val_l1_loss", 9e9)) for r in hist), default="N/A")
    best_ep   = next((r["epoch"] for r in hist
                      if r.get("val_loss", r.get("val_l1_loss", 9e9)) == best_val), "N/A")
    n_ep      = len(hist)
    n_params  = m.get("n_params", "N/A")

    kpis = f"""
<div class="kpi-grid">
  <div class="kpi"><div class="kv">{ds.get('n_train','?')+ds.get('n_val',0) if isinstance(ds.get('n_train'),int) else '?'}</div><div class="kl">Total episodes</div></div>
  <div class="kpi"><div class="kv">{n_params:,}</div><div class="kl">Model params</div></div>
  <div class="kpi"><div class="kv">{n_ep}</div><div class="kl">Epochs trained</div></div>
  <div class="kpi"><div class="kv">{f'{best_val:.4f}' if isinstance(best_val,float) else best_val}</div><div class="kl">Best val loss</div></div>
</div>"""

    ds_params = {
        "Dataset ID":          ds.get("repo_id", "local/aic_cable_insertion"),
        "Episodes":            f"{ds.get('n_train','?')} train  /  {ds.get('n_val','?')} val",
        "Steps per episode":   60,
        "FPS":                 20,
        "Observation: state":  "26-dim float32  (TCP pose 7D + velocity 6D + error 6D + joints 7D)",
        "Observation: image":  "center_camera  128×144 px  RGB  (scales to 256×288 for real AIC)",
        "Action":              "6-dim cartesian twist  [linear.x/y/z, angular.x/y/z]",
        "Action chunk size":   20,
        "Format":              "LeRobot v3.0  (parquet + PNG images)",
        "Note":                "Add left_camera + right_camera at 256×288 for real AIC robot",
    }

    model_params = {
        "Framework":             "LeRobot 0.5.1  (cloned from GitHub)",
        "Policy":                "ACT (Action Chunking with Transformers)",
        "Vision backbone":       "ResNet-18  (randomly initialised for synthetic data)",
        "Transformer dim":       m.get("dim_model", 256),
        "Attention heads":       m.get("n_heads", 8),
        "Encoder layers":        m.get("n_enc_layers", 4),
        "Decoder layers":        m.get("n_dec_layers", 4),
        "CVAE latent dim":       m.get("latent_dim", 32),
        "Chunk size":            m.get("chunk_size", 20),
        "Action steps/pass":     m.get("n_action_steps", 10),
        "Total parameters":      f"{n_params:,}",
    }

    train_params = {
        "Optimiser":             "AdamW",
        "Learning rate (init)":  tr.get("lr", 1e-5),
        "Weight decay":          tr.get("weight_decay", 1e-4),
        "LR schedule":           "Cosine annealing",
        "Steps":                 tr.get("steps", 3000),
        "Batch size":            tr.get("batch_size", 8),
        "KL weight (β)":         tr.get("kl_weight", 10.0),
        "Gradient clipping":     "1.0",
        "Device":                "NVIDIA RTX 4000 Ada  (CUDA)",
    }

    result_params = {
        "Best epoch":            best_ep,
        "Best val loss":         f"{best_val:.5f}" if isinstance(best_val, float) else best_val,
        "Final train loss":      f"{hist[-1].get('train_loss', hist[-1].get('train_l1_loss', 0)):.5f}" if hist else "N/A",
        "Final val loss":        f"{hist[-1].get('val_loss',   hist[-1].get('val_l1_loss',   0)):.5f}" if hist else "N/A",
    }

    arch = """Training path:
  obs (state + image) ──────────────────────────────────────────────────────────────┐
                                                                                     ├─► Transformer ─► CVAE decoder ─► action chunk (20 steps)
  action_seq ──► CVAE encoder ──► z ~ N(μ, σ)  ────────────────────────────────────┘

Inference path:
  obs ──► Transformer encoder ──► CVAE decoder (z = 0) ──► action chunk

Loss:  L = L1(pred, gt) + β · KL( N(μ,σ) ‖ N(0,I) )    β = 10

AIC-specific modifications vs ACT defaults:
  • dim_model  256 (default 512) — smaller for synthetic dataset speed
  • chunk_size  20 (default 100) — shorter horizon for cable-insertion
  • ResNet-18 backbone, no pretrained weights (use ImageNet weights for real data)
  • Single camera input (add left/right cameras at 256×288 for real AIC robot)
  • n_obs_steps=1 (no temporal history — extend for contact-rich insertion)"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>LeRobot ACT Training Report – AIC Robot</title>
<style>{CSS}</style></head>
<body><div class="page">
<h1>LeRobot ACT Training Report – AIC Cable Insertion</h1>
<p class="sub">
  Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
  <span class="tag tb">LeRobot 0.5.1</span> &nbsp;
  <span class="tag tg">ACT</span> &nbsp;
  <span class="tag tp">AIC Robot</span>
</p>

<h2>1. Executive Summary</h2>
{kpis}
<div class="card">{tbl(result_params, "Metric", "Value")}</div>

<h2>2. Dataset</h2>
<div class="card">
  <p style="margin-bottom:1rem;color:#475569">
    Synthetic cable-insertion dataset generated in <strong>LeRobot v3.0 format</strong>
    (parquet + PNG images). Observations match the real AIC robot interface exactly:
    26-dim proprioceptive state (TCP pose, velocity, error, joints) + center camera.
    Left/right cameras omitted here for speed; add them for real-robot training.
  </p>
  {tbl(ds_params)}
</div>
<div class="card">{embed(PLOT_DIR/"04_dataset_overview.png","Dataset overview: TCP pose, joint positions, action distribution, sample frames")}</div>

<h2>3. Model Architecture</h2>
<div class="two">
  <div class="card">{tbl(model_params)}</div>
  <div class="card"><h3>Architecture</h3><pre>{arch}</pre></div>
</div>

<h2>4. Training Method &amp; Hyperparameters</h2>
<div class="card">
  <p style="margin-bottom:1rem;color:#475569">
    Uses <strong>LeRobot's native ACTPolicy</strong> with its built-in normalisation/denormalisation
    pipeline. Training loss = L1 reconstruction + KL divergence (weighted by β).
    During validation, model kept in <code>train()</code> mode to enable VAE encoder for KL computation.
  </p>
  {tbl(train_params)}
</div>

<h2>5. Training Results</h2>
<div class="card">{embed(PLOT_DIR/"01_loss_curves.png","Total training vs validation loss")}</div>
<div class="two">
  <div class="card">{embed(PLOT_DIR/"02_loss_components.png","L1 and KL loss components")}</div>
  <div class="card">{embed(PLOT_DIR/"03_lr_curve.png","Learning-rate schedule")}</div>
</div>

<h2>6. Full Epoch Log</h2>
<div class="card" style="overflow-x:auto">{loss_tbl(hist)}</div>

<h2>7. Using with Real AIC Robot</h2>
<div class="card">
  <p style="margin-bottom:1rem">To adapt this model for real AIC robot data:</p>
  <pre>
# 1. Record real demonstrations with LeRobot
pixi run lerobot-record \\
  --robot.type=aic_controller \\
  --teleop.type=aic_keyboard_ee \\
  --dataset.repo_id=YOUR_HF_USER/aic_real_dataset \\
  --dataset.push_to_hub=false

# 2. Train with all 3 cameras and full 256×288 resolution
#    Update create_aic_dataset.py:
#      N_TRAIN = 100+ real episodes
#      IMG_H, IMG_W = 256, 288
#      Add left_camera + right_camera to features dict

# 3. Update train config:
#      pretrained_backbone_weights = "IMAGENET1K_V1"  (use pretrained ResNet)
#      dim_model = 512  (default ACT size for real data)
#      chunk_size = 100  (standard ACT chunk)
#      lr = 1e-5  (same)

# 4. Load in RunACT.py by pointing to checkpoints_lerobot/best_model</pre>
</div>
</div></body></html>"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "lerobot_training_report.html"
    out.write_text(html)
    print(f"Report saved to {out}")


if __name__ == "__main__":
    build()
