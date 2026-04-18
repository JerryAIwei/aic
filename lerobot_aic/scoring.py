"""
scoring.py — AIC three-tier scoring implementation.

Reference: https://github.com/JerryAIwei/aic/blob/main/docs/scoring.md

Max score per trial: 100 pts
  Tier 1  :   0–1    model validity
  Tier 2  : −36–24   motion quality (smoothness + duration + efficiency + penalties)
  Tier 3  : −12–75   insertion success / partial / proximity
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.signal import savgol_filter as _savgol
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Tier 2: component functions ────────────────────────────────────────────────

def smoothness_score(
    ee_positions: np.ndarray,
    timestamps: np.ndarray,
    speed_threshold: float = 0.01,
) -> float:
    """
    Trajectory smoothness (0–6 pts).

    Metric : time-weighted average linear jerk magnitude (m/s³) via a
             Savitzky–Golay filter (15-sample window, quadratic polynomial
             fit to velocity). Only accumulated when speed > threshold.
    Score  : jerk = 0 m/s³ → 6 pts; jerk ≥ 50 m/s³ → 0 pts; linear between.
    """
    if not _HAS_SCIPY:
        return 0.0

    ee = np.asarray(ee_positions, dtype=float)
    ts = np.asarray(timestamps,   dtype=float)
    if ee.ndim != 2 or ee.shape[1] != 3 or len(ee) < 17:
        return 0.0

    dt = np.diff(ts)
    dt = np.clip(dt, 1e-6, None)
    vel = np.diff(ee, axis=0) / dt[:, None]   # (N-1, 3)
    speeds = np.linalg.norm(vel, axis=1)
    n = len(vel)
    if n < 15:
        return 0.0

    mean_dt = float(np.mean(dt))
    jerk_sq = np.zeros(n)
    for d in range(3):
        j = _savgol(vel[:, d], window_length=15, polyorder=2,
                    deriv=1, delta=mean_dt)
        jerk_sq += j ** 2
    jerk = np.sqrt(jerk_sq)

    moving = speeds > speed_threshold
    if not moving.any():
        return 0.0

    # align arrays (vel has len n, dt has len n for the diffs of ts)
    n_min = min(len(dt), len(jerk), n)
    w = dt[:n_min][moving[:n_min]]
    j = jerk[:n_min][moving[:n_min]]
    if len(w) == 0:
        return 0.0

    mean_jerk = float(np.average(j, weights=w))
    return float(max(0.0, 6.0 * (1.0 - mean_jerk / 50.0)))


def duration_score(duration_s: float) -> float:
    """
    Task duration (0–12 pts).
    ≤ 5 s → 12 pts; ≥ 60 s → 0 pts; linear between.
    """
    if duration_s <= 5.0:
        return 12.0
    if duration_s >= 60.0:
        return 0.0
    return 12.0 * (60.0 - duration_s) / 55.0


def efficiency_score(ee_positions: np.ndarray, initial_distance: float) -> float:
    """
    Trajectory efficiency (0–6 pts).
    Path ≤ initial_distance      → 6 pts (maximum).
    Path ≥ initial_distance + 1m → 0 pts (minimum).
    Linear between.
    """
    ee = np.asarray(ee_positions, dtype=float)
    if ee.ndim != 2 or len(ee) < 2:
        return 0.0
    path = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))
    best  = max(initial_distance, 1e-6)
    worst = best + 1.0
    if path <= best:
        return 6.0
    if path >= worst:
        return 0.0
    return 6.0 * (worst - path) / 1.0


def force_penalty(
    force_magnitudes: np.ndarray,
    timestamps: np.ndarray,
    force_threshold: float = 20.0,
    duration_threshold: float = 1.0,
) -> float:
    """
    Insertion force penalty (0 to −12 pts).
    −12 if force > force_threshold N continuously for > duration_threshold s.
    """
    fm = np.asarray(force_magnitudes, dtype=float)
    ts = np.asarray(timestamps,       dtype=float)
    if len(fm) < 2 or len(ts) < 2:
        return 0.0

    dt = np.diff(ts)
    exceed = fm[:-1] > force_threshold
    max_dur = current = 0.0
    for exc, d in zip(exceed, dt):
        if exc:
            current += float(d)
            if current > max_dur:
                max_dur = current
        else:
            current = 0.0

    return -12.0 if max_dur > duration_threshold else 0.0


def contact_penalty(has_off_limit_contact: bool) -> float:
    """Off-limit contact penalty: −24 if any contact, else 0."""
    return -24.0 if has_off_limit_contact else 0.0


# ── Tier 3 ─────────────────────────────────────────────────────────────────────

def tier3_score(
    insertion_result: str,
    *,
    plug_pos: np.ndarray | None = None,
    port_pos: np.ndarray | None = None,
    initial_plug_port_distance: float | None = None,
    insertion_depth: float | None = None,
    max_insertion_depth: float | None = None,
) -> float:
    """
    Tier 3 – cable insertion.

    insertion_result options:
        'correct'   full insertion into the correct port  → +75
        'wrong'     full insertion into the wrong port    → −12
        'partial'   plug inside port bounding box         → 38–50
        'proximity' plug near port entrance               →  0–25
        'none'      no progress                           →   0
    """
    if insertion_result == "correct":
        return 75.0
    if insertion_result == "wrong":
        return -12.0
    if insertion_result == "partial":
        if insertion_depth is not None and max_insertion_depth:
            ratio = min(1.0, max(0.0, insertion_depth / max_insertion_depth))
            return 38.0 + 12.0 * ratio
        return 38.0
    if insertion_result == "proximity":
        if (plug_pos is not None and port_pos is not None
                and initial_plug_port_distance):
            dist     = float(np.linalg.norm(np.array(plug_pos) - np.array(port_pos)))
            max_dist = initial_plug_port_distance / 2.0
            if dist <= 0:
                return 25.0
            if dist >= max_dist:
                return 0.0
            return 25.0 * (1.0 - dist / max_dist)
        return 0.0
    return 0.0


# ── Main scoring entry point ───────────────────────────────────────────────────

def compute_trial_score(trial: dict) -> dict:
    """
    Compute the full AIC score for a single trial.

    Parameters
    ----------
    trial : dict with the following keys (all optional; missing → 0 contribution):
        valid                    bool    model loaded and ran without error
        ee_positions             (N,3)   end-effector positions (m)
        timestamps               (N,)    timestamps for ee_positions (s)
        duration_s               float   task duration (s)
        initial_plug_port_dist   float   initial plug-to-port distance (m)
        force_magnitudes         (M,)    force sensor readings (N)
        force_timestamps         (M,)    timestamps for force readings (s)
        has_off_limit_contact    bool    any off-limit contact detected
        insertion_result         str     'correct'|'wrong'|'partial'|'proximity'|'none'
        plug_pos                 (3,)    final plug position (for proximity)
        port_pos                 (3,)    target port position (for proximity)
        insertion_depth          float   depth into port (for partial)
        max_insertion_depth      float   full port depth (for partial)

    Returns
    -------
    dict with keys:
        tier1, tier2, tier3, total      floats
        smoothness, duration_score, efficiency, force_penalty, contact_penalty
        available_metrics               list[str]  which metrics were computed
    """
    t1 = 1.0 if trial.get("valid", True) else 0.0

    ins = trial.get("insertion_result", "none")
    t3  = tier3_score(
        ins,
        plug_pos=trial.get("plug_pos"),
        port_pos=trial.get("port_pos"),
        initial_plug_port_distance=trial.get("initial_plug_port_dist"),
        insertion_depth=trial.get("insertion_depth"),
        max_insertion_depth=trial.get("max_insertion_depth"),
    )

    def _to_arr(v):
        if v is None:
            return np.array([], dtype=float)
        a = np.asarray(v, dtype=float)
        return a if a.size > 0 else np.array([], dtype=float)

    ee  = _to_arr(trial.get("ee_positions"))
    ts  = _to_arr(trial.get("timestamps"))
    d_s = float(trial.get("duration_s") or 60.0)
    ipd = float(trial.get("initial_plug_port_dist") or 1.0)
    fm  = _to_arr(trial.get("force_magnitudes"))
    ft  = _to_arr(trial.get("force_timestamps"))
    ofl = bool(trial.get("has_off_limit_contact", False))

    available = []

    # Tier 2 positives only if Tier 3 > 0
    if t3 > 0:
        sm  = smoothness_score(ee, ts) if (ee.ndim == 2 and len(ee) >= 17) else None
        dur = duration_score(d_s) if trial.get("duration_s") is not None else None
        eff = efficiency_score(ee, ipd) if (ee.ndim == 2 and len(ee) >= 2) else None
        if sm  is not None: available.append("smoothness")
        if dur is not None: available.append("duration")
        if eff is not None: available.append("efficiency")
    else:
        sm = dur = eff = None

    fp = force_penalty(fm, ft) if len(fm) >= 2 else None
    cp = contact_penalty(ofl) if trial.get("has_off_limit_contact") is not None else None
    if fp is not None: available.append("force_penalty")
    if cp is not None: available.append("contact_penalty")

    sm_v  = sm  if sm  is not None else 0.0
    dur_v = dur if dur is not None else 0.0
    eff_v = eff if eff is not None else 0.0
    fp_v  = fp  if fp  is not None else 0.0
    cp_v  = cp  if cp  is not None else 0.0

    t2 = sm_v + dur_v + eff_v + fp_v + cp_v

    return {
        "tier1":             round(t1,    4),
        "tier2":             round(t2,    4),
        "tier3":             round(t3,    4),
        "total":             round(t1 + t2 + t3, 4),
        "smoothness":        round(sm_v,  4) if sm  is not None else None,
        "duration_score":    round(dur_v, 4) if dur is not None else None,
        "efficiency":        round(eff_v, 4) if eff is not None else None,
        "force_penalty":     round(fp_v,  4) if fp  is not None else None,
        "contact_penalty":   round(cp_v,  4) if cp  is not None else None,
        "available_metrics": available,
    }


def summarize_scores(trial_scores: list[dict]) -> dict:
    """Aggregate per-trial score dicts into a summary."""
    n = len(trial_scores)
    if n == 0:
        return {"n_trials": 0}

    summary: dict = {"n_trials": n}
    for k in ("total", "tier1", "tier2", "tier3",
              "smoothness", "duration_score", "efficiency",
              "force_penalty", "contact_penalty"):
        vals = [t[k] for t in trial_scores if t.get(k) is not None]
        if vals:
            summary[f"mean_{k}"] = round(float(np.mean(vals)), 4)
            summary[f"std_{k}"]  = round(float(np.std(vals)),  4)

    summary["n_success"]    = sum(1 for t in trial_scores if (t.get("tier3") or 0) >= 75)
    summary["success_rate"] = round(summary["n_success"] / n, 4)
    return summary
