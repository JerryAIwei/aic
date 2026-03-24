"""
compare_act_diffusion.py
Loads training metrics from ACT and Diffusion Policy runs and prints
a side-by-side comparison report.

Usage:
    python compare_act_diffusion.py
"""

import json
from pathlib import Path

ACT_METRICS   = Path(__file__).parent / "outputs_act"   / "metrics.json"
DIFF_METRICS  = Path(__file__).parent / "outputs_diffusion" / "metrics.json"
ACT_CFG       = Path(__file__).parent / "outputs_act"   / "run_config.json"
DIFF_CFG      = Path(__file__).parent / "outputs_diffusion" / "run_config.json"


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found: {path}\nRun the trainer first.")
    with open(path) as f:
        return json.load(f)


def load_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def best_val(records: list[dict], key: str = "val_loss") -> tuple[float, int]:
    bv = float("inf")
    be = 0
    for r in records:
        v = r.get(key) or r.get("val_loss") or float("inf")
        if v < bv:
            bv, be = v, r.get("epoch", 0)
    return bv, be


def summarise(name: str, records: list[dict], cfg: dict) -> dict:
    key = "val_loss"
    train_key = "train_loss"

    bv, be = best_val(records, key)
    final = records[-1]
    n_epochs = final.get("epoch", len(records))
    n_steps  = final.get("step",  0)

    # Convergence: epoch where val loss first drops below 110 % of best val
    conv_epoch = n_epochs
    thresh = bv * 1.10
    for r in records:
        v = r.get(key) or float("inf")
        if v <= thresh:
            conv_epoch = r.get("epoch", 0)
            break

    return {
        "policy":         name,
        "n_params":       cfg.get("model", {}).get("n_params", "?"),
        "n_epochs":       n_epochs,
        "n_steps":        n_steps,
        "final_train":    final.get(train_key, float("nan")),
        "final_val":      final.get(key,       float("nan")),
        "best_val":       bv,
        "best_val_epoch": be,
        "conv_epoch":     conv_epoch,
    }


def print_report(act_s: dict, diff_s: dict, act_r: list, diff_r: list):
    w = 42
    sep = "─" * (w * 2 + 5)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.5f}" if not (v != v) else "N/A"
        return str(v)

    print("\n" + "=" * (w * 2 + 5))
    print(f"  ACT vs Diffusion Policy — Training Comparison")
    print("=" * (w * 2 + 5))

    rows = [
        ("Policy",                 "act_policy",    "diff_policy"),
        ("Parameters",             "n_params",       "n_params"),
        ("Epochs trained",         "n_epochs",       "n_epochs"),
        ("Gradient steps",         "n_steps",        "n_steps"),
        ("Final train loss",       "final_train",    "final_train"),
        ("Final val loss",         "final_val",      "final_val"),
        ("Best val loss",          "best_val",       "best_val"),
        ("Best val epoch",         "best_val_epoch", "best_val_epoch"),
        ("Convergence epoch*",     "conv_epoch",     "conv_epoch"),
    ]

    act_s["act_policy"]  = act_s["policy"]
    diff_s["diff_policy"] = diff_s["policy"]

    header = f"{'Metric':<30}  {'ACT':>{w-32}}  {'Diffusion':>{w-32}}"
    print(f"\n{header}")
    print(sep)
    for label, ak, dk in rows:
        av = fmt(act_s.get(ak, "?"))
        dv = fmt(diff_s.get(dk, "?"))
        print(f"  {label:<28}  {av:>{w-32}}  {dv:>{w-32}}")

    print(sep)
    print("  * Convergence epoch: first epoch ≤ 110 % of best val loss")

    # Per-epoch table
    print("\n\n  Per-epoch Loss Curves")
    print(f"  {'Epoch':>5}  {'ACT train':>10}  {'ACT val':>9}  {'Diff train':>10}  {'Diff val':>9}")
    print("  " + "─" * 50)

    # Align by epoch
    act_by_ep  = {r["epoch"]: r for r in act_r}
    diff_by_ep = {r["epoch"]: r for r in diff_r}
    all_epochs = sorted(set(act_by_ep) | set(diff_by_ep))

    # Print at most 30 rows (sample uniformly)
    step = max(1, len(all_epochs) // 30)
    for ep in all_epochs[::step]:
        ar = act_by_ep.get(ep,  {})
        dr = diff_by_ep.get(ep, {})
        at = ar.get("train_loss", float("nan"))
        av = ar.get("val_loss",   float("nan"))
        dt = dr.get("train_loss", float("nan"))
        dv = dr.get("val_loss",   float("nan"))
        print(f"  {ep:>5}  {at:>10.5f}  {av:>9.5f}  {dt:>10.5f}  {dv:>9.5f}")

    print()

    # Verdict
    print("  Verdict")
    print("  " + "─" * 50)
    if act_s["best_val"] < diff_s["best_val"]:
        winner = "ACT"
        margin = (diff_s["best_val"] - act_s["best_val"]) / diff_s["best_val"] * 100
    elif diff_s["best_val"] < act_s["best_val"]:
        winner = "Diffusion Policy"
        margin = (act_s["best_val"] - diff_s["best_val"]) / act_s["best_val"] * 100
    else:
        winner = "Tie"
        margin = 0.0

    print(f"  Best val loss winner : {winner} ({margin:.1f} % better)")
    print(f"  ACT best val loss    : {act_s['best_val']:.5f}  (epoch {act_s['best_val_epoch']})")
    print(f"  Diffusion best val   : {diff_s['best_val']:.5f}  (epoch {diff_s['best_val_epoch']})")
    print()

    print("  Notes")
    print("  ─────────────────────────────────────────────")
    print("  • Both trained on local/aic_cable_insertion_large (synthetic data)")
    print("  • Diffusion Policy uses DDPM denoising — val loss is noise-pred MSE")
    print("  • ACT uses CVAE — val loss = action reconstruction + KL term")
    print("  • Lower val loss within each model is meaningful; cross-model")
    print("    comparison is qualitative (different loss scales)")
    print()


def main():
    act_r  = load(ACT_METRICS)
    diff_r = load(DIFF_METRICS)
    act_cfg  = load_cfg(ACT_CFG)
    diff_cfg = load_cfg(DIFF_CFG)

    act_s  = summarise("ACT",              act_r,  act_cfg)
    diff_s = summarise("Diffusion Policy", diff_r, diff_cfg)

    print_report(act_s, diff_s, act_r, diff_r)

    # Save report JSON
    report_path = Path(__file__).parent / "outputs_act" / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump({"act": act_s, "diffusion": diff_s}, f, indent=2)
    print(f"  Report saved to {report_path}")


if __name__ == "__main__":
    main()
