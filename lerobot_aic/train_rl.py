"""
train_rl.py
RL fine-tuning of a behavior-cloned MLP policy for cable insertion.

Algorithm:  PPO initialized from BC pre-training on iter10 demonstration data.
Reward:     dense exp(-20 * dist_to_port) — peaks at 0.5 when at port, floors at -0.5.
Port pos:   estimated as mean final TCP position across successful BC demonstrations.

Pipeline:
  Phase 1 — BC pre-train:  fit MLP (state → action) on existing npz episodes.
  Phase 2 — RL loop:       for each iteration:
              a) publish weights → /tmp/mlp_rl/weights.pt
              b) run one Gazebo episode with RunMLPPolicy (saves trajectory npz)
              c) compute per-step rewards from TCP→port distance
              d) PPO update on full episode trajectory
              e) checkpoint every 5 iterations

Usage:
    # Full run (BC + RL)
    python train_rl.py

    # Skip BC, load existing weights and continue RL
    python train_rl.py --skip_bc --rl_iter 20

    # BC only (use for initial test before RL loop)
    python train_rl.py --rl_iter 0
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ── Constants ─────────────────────────────────────────────────────────────────
SAVE_DIR  = Path("/tmp/aic_recordings")
MLP_DIR   = Path("/tmp/mlp_rl")
ROOTFS    = "/opt/aic_rootfs"
ROS_SETUP = "/ws_aic/install/setup.bash"
STATE_DIM_FULL = 26   # full observation vector
STATE_DIM  = 14       # robust features only: TCP pose (7) + joint positions (7)
ACTION_DIM = 6
FPS = 20

# Indices into the 26-D observation that are ROBUST (same semantics in demo and RL env):
#   0-6:  TCP pose (xyz + quaternion) — always the actual TCP pose, no mismatch
#   19-25: joint positions            — always the actual joint angles, no mismatch
# Excluded (semantic mismatch between CheatCode demos and MLP policy RL env):
#   7-12:  tcp_velocity               — differs (CheatCode drives robot actively; MLP is slower)
#   13-18: tcp_error                  — CRITICAL MISMATCH: CheatCode keeps error at ~3.2cm below;
#                                       MLP policy keeps error near 0 (target=current TCP)
_ROBUST_IDX = list(range(7)) + list(range(19, 26))   # 14 features

# Standard scenario config (fixed board, no variation) for reproducible RL rollouts
_FIXED_TRIAL_CONFIG = """# Fixed-scenario config for RL training
scoring:
  topics:
    - topic:
        name: "/joint_states"
        type: "sensor_msgs/msg/JointState"
    - topic:
        name: "/tf"
        type: "tf2_msgs/msg/TFMessage"
    - topic:
        name: "/tf_static"
        type: "tf2_msgs/msg/TFMessage"
        latched: true
    - topic:
        name: "/scoring/tf"
        type: "tf2_msgs/msg/TFMessage"
    - topic:
        name: "/aic/gazebo/contacts/off_limit"
        type: "ros_gz_interfaces/msg/Contacts"
    - topic:
        name: "/fts_broadcaster/wrench"
        type: "geometry_msgs/msg/WrenchStamped"
    - topic:
        name: "/aic_controller/joint_commands"
        type: "aic_control_interfaces/msg/JointMotionUpdate"
    - topic:
        name: "/aic_controller/pose_commands"
        type: "aic_control_interfaces/msg/MotionUpdate"
    - topic:
        name: "/scoring/insertion_event"
        type: "std_msgs/msg/String"
    - topic:
        name: "/aic_controller/controller_state"
        type: "aic_control_interfaces/msg/ControllerState"

task_board_limits:
  nic_rail:
    min_translation: -0.0215
    max_translation: 0.0234
  sc_rail:
    min_translation: -0.06
    max_translation: 0.055
  mount_rail:
    min_translation: -0.09425
    max_translation: 0.09425

trials:
  trial_1:
    scene:
        task_board:
          pose:
            x: 0.15
            y: -0.20
            z: 1.14
            roll: 0.0
            pitch: 0.0
            yaw: 3.1415
          nic_rail_0:
            entity_present: True
            entity_name: "nic_card_0"
            entity_pose:
              translation: 0.036
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
          nic_rail_1:
            entity_present: False
          nic_rail_2:
            entity_present: False
          nic_rail_3:
            entity_present: False
          nic_rail_4:
            entity_present: False
          sc_rail_0:
            entity_present: True
            entity_name: "sc_mount_0"
            entity_pose:
              translation: 0.042
              roll: 0.0
              pitch: 0.0
              yaw: 0.1
          sc_rail_1:
            entity_present: False
          lc_mount_rail_0:
            entity_present: True
            entity_name: "lc_mount_0"
            entity_pose:
              translation: 0.02
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
          sfp_mount_rail_0:
            entity_present: True
            entity_name: "sfp_mount_0"
            entity_pose:
              translation: 0.03
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
          sc_mount_rail_0:
            entity_present: True
            entity_name: "sc_mount_0"
            entity_pose:
              translation: -0.02
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
          lc_mount_rail_1:
            entity_present: True
            entity_name: "lc_mount_1"
            entity_pose:
              translation: -0.01
              roll: 0.0
              pitch: 0.0
              yaw: 0.0
          sfp_mount_rail_1:
            entity_present: False
          sc_mount_rail_1:
            entity_present: False
        cables:
          cable_0:
            pose:
              gripper_offset:
                x: 0.0
                y: 0.015385
                z: 0.04245
              roll: 0.4432
              pitch: -0.4838
              yaw: 1.3303
            attach_cable_to_gripper: True
            cable_type: "sfp_sc_cable"
    tasks:
      task_1:
        cable_type: "sfp_sc"
        cable_name: "cable_0"
        plug_type: "sfp"
        plug_name: "sfp_tip"
        port_type: "sfp"
        port_name: "sfp_port_0"
        target_module_name: "nic_card_mount_0"
        time_limit: 180

robot:
  home_joint_positions:
    shoulder_pan_joint: -0.1597
    shoulder_lift_joint: -1.3542
    elbow_joint: -1.6648
    wrist_1_joint: -1.6933
    wrist_2_joint: 1.5710
    wrist_3_joint: 1.4110
"""


# ── Actor-Critic Network ───────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Shared-trunk actor-critic for state-only MLP policy.

    Inputs:  normalized 26-D state (TCP pose 7D + velocity 6D + tcp_error 6D + joints 7D)
    Outputs: action mean (6-D), action std (6-D), value scalar
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -1.5))
        self.critic        = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        h   = self.trunk(state)
        mean = self.actor_mean(h)
        std  = self.actor_log_std.exp().expand_as(mean)
        val  = self.critic(h).squeeze(-1)
        return mean, std, val


# ── BC Pre-training ────────────────────────────────────────────────────────────

def load_npz_episodes(npz_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all ep_*.npz files. Returns (robust_states, actions, port_pos_estimate).

    robust_states uses only _ROBUST_IDX features (TCP pose + joints, 14D)
    to avoid the tcp_velocity / tcp_error mismatch between demo and RL environments.

    port_pos_estimate is the mean TCP position over the last 20 frames of each
    successful episode — a good proxy for the insertion port location.
    """
    files = sorted(npz_dir.glob("ep_*.npz"))
    if not files:
        raise FileNotFoundError(f"No ep_*.npz found in {npz_dir}")

    all_states, all_actions, final_tcps = [], [], []
    for f in files:
        d = np.load(str(f))
        if "states" not in d or len(d["states"]) < 10:
            continue
        # Select only the robust feature indices (14D)
        all_states.append(d["states"][:, _ROBUST_IDX].astype(np.float32))
        all_actions.append(d["actions"].astype(np.float32))
        final_tcps.append(d["states"][-20:, :3].mean(0))  # xyz mean of last 20 frames

    if not all_states:
        raise RuntimeError("All npz files are empty or invalid.")

    states   = np.concatenate(all_states,  axis=0)   # (N, 14) robust only
    actions  = np.concatenate(all_actions, axis=0)   # (N,  6)
    port_pos = np.array(final_tcps).mean(0)          # (3,) estimated port xyz

    print(f"Loaded {len(files)} episodes → {len(states):,} steps")
    print(f"Robust state dim: {states.shape[1]}  (features: TCP-pose 7D + joints 7D)")
    print(f"Port position estimate (mean final TCP): {port_pos.round(4)}")
    return states, actions, port_pos


def compute_normalization(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0).astype(np.float32)
    std  = (states.std(0) + 1e-6).astype(np.float32)
    return mean, std


def bc_pretrain(model: ActorCritic, states: np.ndarray, actions: np.ndarray,
                port_pos: np.ndarray, state_mean: np.ndarray, state_std: np.ndarray,
                steps: int = 2000, batch_size: int = 256, lr: float = 3e-4,
                gamma: float = 0.99,
                device: torch.device = torch.device("cpu")) -> None:
    """Behavioral cloning + critic pre-training.

    Phase A: MSE actor loss on demonstrated actions.
    Phase B: Critic pre-training — predict the potential-based return from BC trajectories.
    Pre-training the critic prevents huge VF loss during early RL iterations.
    """
    # ── Phase A: Actor MSE ────────────────────────────────────────────────────
    model.train()
    actor_params  = list(model.trunk.parameters()) + list(model.actor_mean.parameters())
    actor_opt     = Adam(actor_params, lr=lr, weight_decay=1e-5)

    states_t  = torch.from_numpy((states  - state_mean) / state_std).float().to(device)
    actions_t = torch.from_numpy(actions).float().to(device)
    N = len(states_t)

    print(f"\n── BC Phase A: Actor MSE ({steps} steps, batch={batch_size}) ──")
    best_loss = float("inf")
    for step in range(steps):
        idx   = torch.randint(0, N, (batch_size,))
        s_b, a_b = states_t[idx], actions_t[idx]

        mean, _, _ = model(s_b)
        loss = F.mse_loss(mean, a_b)

        actor_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        actor_opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1:5d}/{steps}  loss={loss.item():.6f}  best={best_loss:.6f}")

    print(f"BC Phase A done. Best actor loss: {best_loss:.6f}")

    # ── Phase B: Critic pre-training on BC trajectory returns ─────────────────
    # Compute potential-based returns for each BC trajectory (reloaded from files)
    print(f"\n── BC Phase B: Critic pre-train ──")
    # Only update the critic head — keep trunk frozen so Phase A actor weights are preserved
    critic_opt = Adam(model.critic.parameters(), lr=lr, weight_decay=1e-5)

    # Build (state, return) pairs from BC episodes
    # Split states/actions back into per-episode chunks using action discontinuities
    # (simpler: just use random 400-step windows as pseudo-episodes)
    chunk = 400
    all_ret = []
    for start in range(0, N - chunk, chunk):
        ep_s = states[start: start + chunk]
        _, ep_ret = compute_rewards_and_returns(ep_s, port_pos, gamma=gamma)
        all_ret.append(ep_ret)
    bc_returns = np.concatenate(all_ret, axis=0)
    bc_states  = np.concatenate([states[i: i + chunk]
                                 for i in range(0, N - chunk, chunk)], axis=0)
    R = len(bc_states)
    bc_s_t   = torch.from_numpy((bc_states  - state_mean) / state_std).float().to(device)
    bc_ret_t = torch.from_numpy(bc_returns).float().to(device)

    for step in range(1000):
        idx = torch.randint(0, R, (batch_size,))
        _, _, val = model(bc_s_t[idx])
        v_loss = F.mse_loss(val, bc_ret_t[idx])

        critic_opt.zero_grad()
        v_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        critic_opt.step()

        if (step + 1) % 250 == 0:
            print(f"  critic step {step+1:4d}/1000  v_loss={v_loss.item():.4f}")

    print(f"BC Phase B done.")
    model.eval()


# ── Reward & Return computation ────────────────────────────────────────────────

TARGET_Z = 0.1905   # port height (consistent across ALL scenarios — z doesn't change with board XY)


def compute_rewards_and_returns(
    states: np.ndarray, port_pos: np.ndarray, gamma: float = 0.99
) -> tuple[np.ndarray, np.ndarray]:
    """Mixed reward: z-approach (reliable) + 3D proximity bonus.

    Primary reward — z-axis approach (highly reliable: port z=0.190 is the same
    for ALL board configurations regardless of XY variation):
        r_z_t = (|z_{t-1} - TARGET_Z| - |z_t - TARGET_Z|) * 200
        Positive when moving toward target height, negative when moving away.

    Secondary reward — 3D proximity bonus (softer, using mean port pos estimate):
        r_prox_t = exp(-10 * ||tcp_t - port_pos||) * 0.3
        Provides a continuous incentive to stay close to the port.

    Combined: policy learns to go DOWN to z=0.190 while staying near (x,y)≈port_pos.
    """
    tcp_pos = states[:, :3]                                    # (T, 3)
    z_pos   = tcp_pos[:, 2]                                    # (T,) z coord

    # Z-approach reward (primary, reliable)
    z_dist  = np.abs(z_pos - TARGET_Z)                        # (T,) distance from target z
    z_delta = np.diff(z_dist, prepend=z_dist[0])              # (T,) Δz_dist (negative = good)
    r_z     = -z_delta * 200.0                                # scale: 1cm z-approach = +2.0

    # 3D proximity bonus (secondary, informative)
    dists_3d = np.linalg.norm(tcp_pos - port_pos, axis=1)     # (T,)
    r_prox   = np.exp(-10.0 * dists_3d) * 0.3                # peaks at 0.3 when at port

    rewards = r_z + r_prox                                     # (T,)

    # Discounted returns (backwards pass)
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G = 0.0
    for t in reversed(range(T)):
        G = float(rewards[t]) + gamma * G
        returns[t] = G

    return rewards.astype(np.float32), returns


# ── PPO Update ────────────────────────────────────────────────────────────────

def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,        # (T, 26) normalized
    actions: torch.Tensor,       # (T, 6)
    old_log_probs: torch.Tensor, # (T,)
    returns: torch.Tensor,       # (T,) discounted
    bc_model: ActorCritic | None = None,  # frozen BC reference (for KL constraint)
    n_epochs: int = 1,
    batch_size: int = 512,
    clip_eps: float = 0.05,     # very conservative: 5% probability ratio change
    vf_coef: float = 0.5,
    kl_coef: float = 0.5,       # KL penalty weight vs BC reference policy
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Conservative PPO update with BC-KL regularization.

    Unlike standard PPO, this version:
    1. Uses a tight clip (0.05) to prevent large policy changes per iteration.
    2. Freezes actor_log_std (no entropy tuning) to keep action std stable.
    3. Adds a KL divergence penalty from the reference BC policy to prevent drift.
       This mirrors RLHF's reference-policy constraint.
    4. Single epoch over the trajectory (n_epochs=1 default).

    The KL term: E[KL(current || bc)] keeps the RL policy close to the BC prior
    while allowing the policy gradient to push it toward higher reward.
    """
    T = len(states)

    with torch.no_grad():
        _, _, values = model(states)
        if bc_model is not None:
            bc_mean, bc_std, _ = bc_model(states)

    advantages = (returns - values).detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    agg = {"policy_loss": 0.0, "value_loss": 0.0, "kl_bc": 0.0}
    n = 0

    # Freeze actor_log_std — do NOT allow RL to change action std
    # (prevents entropy from exploding and making actions erratic)
    saved_std_grad = model.actor_log_std.requires_grad
    model.actor_log_std.requires_grad_(False)

    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(T, device=device)
        for start in range(0, T, batch_size):
            idx = perm[start: start + batch_size]
            if len(idx) < 2:
                continue

            s_b   = states[idx]
            a_b   = actions[idx]
            adv_b = advantages[idx]
            ret_b = returns[idx]
            olp_b = old_log_probs[idx]

            mean, std, val = model(s_b)
            dist   = torch.distributions.Normal(mean, std)
            new_lp = dist.log_prob(a_b).sum(-1)

            ratio = torch.exp(new_lp - olp_b)
            surr  = torch.min(
                ratio * adv_b,
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_b,
            )
            p_loss = -surr.mean()
            v_loss = F.mse_loss(val, ret_b)

            # BC-KL regularization: penalize drift from original BC policy
            kl_loss = torch.zeros(1, device=device)
            if bc_model is not None:
                bc_m_b = bc_mean[idx].detach()
                bc_s_b = bc_std[idx].detach()
                # KL(current || bc): symmetric approximate form
                var_ratio = (std / bc_s_b) ** 2
                kl_loss   = 0.5 * (var_ratio + ((mean - bc_m_b) / bc_s_b) ** 2
                                   - 1 - torch.log(var_ratio + 1e-8)).sum(-1).mean()

            loss = p_loss + vf_coef * v_loss + kl_coef * kl_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            agg["policy_loss"] += p_loss.item()
            agg["value_loss"]  += v_loss.item()
            agg["kl_bc"]       += kl_loss.item()
            n += 1

    model.actor_log_std.requires_grad_(saved_std_grad)
    model.eval()
    return {k: v / max(n, 1) for k, v in agg.items()}


# ── Subprocess / Simulation helpers ───────────────────────────────────────────

def _ros_env() -> dict:
    env = os.environ.copy()
    rootfs_libs = (
        f"{ROOTFS}/usr/lib/x86_64-linux-gnu:"
        f"{ROOTFS}/lib/x86_64-linux-gnu:"
        f"{ROOTFS}/usr/lib"
    )
    ogre      = f"{ROOTFS}/usr/lib/x86_64-linux-gnu/OGRE-2.3"
    gz_vendor = f"{ROOTFS}/opt/ros/kilted/opt/gz_ogre_next_vendor/lib"
    sys_libs  = "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
    existing  = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"]            = f"{sys_libs}:{rootfs_libs}:{ogre}:{gz_vendor}:{existing}"
    env["RMW_IMPLEMENTATION"]         = "rmw_zenoh_cpp"
    env["ZENOH_ROUTER_CHECK_ATTEMPTS"] = "-1"
    env["ZENOH_CONFIG_OVERRIDE"]      = "transport/shared_memory/enabled=false"
    env.pop("DISPLAY", None)
    rootfs_py = f"{ROOTFS}/usr/lib/python3/dist-packages:{ROOTFS}/usr/lib/python3.12/dist-packages"
    env["PYTHONPATH"] = f"{rootfs_py}:{env.get('PYTHONPATH', '')}"
    return env


def _popen(cmd: str, log: Path | None = None, env: dict | None = None) -> subprocess.Popen:
    if log:
        cmd = f"({cmd}) 2>&1 | tee {log}"
    return subprocess.Popen(
        ["bash", "-c", f". {ROS_SETUP} && {cmd}"],
        env=env or _ros_env(),
        start_new_session=True,
    )


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _cleanup_stale_nodes() -> None:
    for pattern in [
        "aic_model", "rmw_zenohd", "gz_server", "gzserver",
        "component_container", "aic_engine", "aic_adapter",
        "aic_bringup", "robot_state_pub", "ros_gz", "spawner",
        "ros2 launch", "gz sim",
    ]:
        subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
    time.sleep(4)


def _wait_for_episode(timeout: int = 240) -> Path | None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    latest = SAVE_DIR / "latest.txt"
    latest.unlink(missing_ok=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if latest.exists():
            text = latest.read_text().strip()
            if text:
                ep_path = Path(text)
                if ep_path.exists() and ep_path.suffix == ".npz":
                    return ep_path
        time.sleep(2)
    return None


def run_rl_episode(ep_idx: int) -> Path | None:
    """Spin up one fixed-scenario Gazebo episode with RunMLPPolicy.
    Returns path to saved trajectory npz, or None on timeout.
    """
    _cleanup_stale_nodes()

    cfg_path = Path("/tmp/aic_rl_trial.yaml")
    cfg_path.write_text(_FIXED_TRIAL_CONFIG)
    log_dir = Path("/tmp/aic_logs_rl")
    log_dir.mkdir(exist_ok=True)
    env = _ros_env()

    zenoh = _popen(
        "ros2 run rmw_zenoh_cpp rmw_zenohd",
        log=log_dir / f"zenoh_{ep_idx}.log", env=env,
    )
    time.sleep(3)

    policy = _popen(
        "ros2 run aic_model aic_model "
        "--ros-args -p use_sim_time:=true "
        "-p policy:=aic_example_policies.ros.RunMLPPolicy",
        log=log_dir / f"policy_{ep_idx}.log", env=env,
    )
    time.sleep(2)

    sim = _popen(
        "ros2 launch aic_bringup aic_gz_bringup.launch.py "
        "gazebo_gui:=false "
        "ground_truth:=true "
        "spawn_task_board:=false "
        "start_aic_engine:=true "
        "shutdown_on_aic_engine_exit:=true "
        f"aic_engine_config_file:={cfg_path}",
        log=log_dir / f"sim_{ep_idx}.log", env=env,
    )
    time.sleep(50)  # wait for Gazebo + controller to fully start

    ep_path = _wait_for_episode(timeout=280)

    for proc in [policy, sim, zenoh]:
        _kill(proc)
    time.sleep(3)

    return ep_path


# ── Persistence helpers ────────────────────────────────────────────────────────

def publish_weights(model: ActorCritic) -> None:
    """Write actor-critic weights where RunMLPPolicy can read them."""
    MLP_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MLP_DIR / "weights.pt")


def save_config(state_mean: np.ndarray, state_std: np.ndarray,
                port_pos: np.ndarray) -> None:
    MLP_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "state_mean": state_mean.tolist(),
        "state_std":  state_std.tolist(),
        "port_pos":   port_pos.tolist(),
    }
    json.dump(config, open(MLP_DIR / "config.json", "w"), indent=2)


def save_checkpoint(model: ActorCritic, optimizer, iteration: int,
                    state_mean: np.ndarray, state_std: np.ndarray,
                    port_pos: np.ndarray, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "iteration":  iteration,
        "state_mean": state_mean,
        "state_std":  state_std,
        "port_pos":   port_pos,
    }, ckpt_dir / f"ckpt_{iteration:03d}.pt")
    publish_weights(model)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BC+PPO RL fine-tuning for cable insertion")
    p.add_argument("--npz_dir",    default="/tmp/aic_recordings_iter10_backup",
                   help="Directory with BC demonstration npz files")
    p.add_argument("--bc_steps",   type=int,   default=3000,
                   help="BC pre-training gradient steps")
    p.add_argument("--rl_iter",    type=int,   default=30,
                   help="Number of RL episodes to collect and train on")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--skip_bc",    action="store_true",
                   help="Load existing /tmp/mlp_rl/weights.pt instead of BC pre-training")
    p.add_argument("--run_name",   default="rl_v1")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_dir = Path(__file__).parent / f"checkpoints_rl_{args.run_name}"
    out_dir  = Path(__file__).parent / f"outputs_rl_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    MLP_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load BC data ───────────────────────────────────────────────────────
    npz_dir = Path(args.npz_dir)
    states, actions, port_pos = load_npz_episodes(npz_dir)
    state_mean, state_std = compute_normalization(states)
    print(f"State normalization from {len(states):,} BC frames")

    # ── 2. Build model ────────────────────────────────────────────────────────
    model     = ActorCritic(STATE_DIM, ACTION_DIM, hidden=256).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # BC reference model (frozen copy used for KL regularization)
    bc_ref_model = ActorCritic(STATE_DIM, ACTION_DIM, hidden=256).to(device)

    # ── 3. BC pre-train (or load) ─────────────────────────────────────────────
    weights_path = MLP_DIR / "weights.pt"
    if args.skip_bc and weights_path.exists():
        print(f"Loading existing weights from {weights_path}")
        ckpt = torch.load(str(weights_path), map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        model.eval()
    else:
        bc_pretrain(model, states, actions, port_pos, state_mean, state_std,
                    steps=args.bc_steps, gamma=args.gamma, device=device)
        save_checkpoint(model, optimizer, 0, state_mean, state_std, port_pos, ckpt_dir)

    # Freeze a copy of the BC weights as the reference policy
    bc_ref_model.load_state_dict(model.state_dict())
    bc_ref_model.eval()
    for p in bc_ref_model.parameters():
        p.requires_grad_(False)

    # Write config for RunMLPPolicy
    save_config(state_mean, state_std, port_pos)
    publish_weights(model)
    print(f"\nConfig + weights written to {MLP_DIR}")
    print(f"Port position estimate: {port_pos.round(4)}")

    if args.rl_iter == 0:
        print("rl_iter=0: BC-only run complete.")
        return

    # ── 4. RL loop ────────────────────────────────────────────────────────────
    history = []
    print(f"\n── RL Fine-tuning ({args.rl_iter} iterations) ──")
    print(f"{'Iter':>5}  {'Steps':>6}  {'ZAppr':>6}  {'ZFin':>7}  "
          f"{'PLoss':>8}  {'VLoss':>8}  {'KL_BC':>7}  {'Elapsed':>7}")
    print("─" * 70)

    for iteration in range(1, args.rl_iter + 1):
        t0 = time.time()

        # a. Publish current weights before launching episode
        publish_weights(model)

        # b. Run one episode in sim with RunMLPPolicy
        print(f"  [RL {iteration}/{args.rl_iter}] Launching episode …", flush=True)
        ep_path = run_rl_episode(iteration)
        if ep_path is None:
            print(f"{iteration:>5}  TIMEOUT — skipping")
            continue

        # c. Load trajectory (states-only npz saved by RunMLPPolicy)
        d              = np.load(str(ep_path))
        ep_states_full = d["states"].astype(np.float32)   # (T, 26) full state
        ep_states      = ep_states_full[:, _ROBUST_IDX]   # (T, 14) robust features only
        ep_actions     = d["actions"].astype(np.float32)  # (T, 6)
        T = len(ep_states)

        # d. Compute rewards and returns (uses full state for TCP xyz in indices 0-2)
        rewards, returns = compute_rewards_and_returns(ep_states_full, port_pos, gamma=args.gamma)
        final_dist  = float(np.linalg.norm(ep_states_full[-1, :3] - port_pos))
        mean_reward = float(rewards.mean())

        # e. Build tensors
        s_norm = (ep_states - state_mean) / (state_std + 1e-8)
        s_t    = torch.from_numpy(s_norm).float().to(device)
        a_t    = torch.from_numpy(ep_actions).float().to(device)
        ret_t  = torch.from_numpy(returns).float().to(device)

        with torch.no_grad():
            mean_pred, std_pred, _ = model(s_t)
            dist_pred = torch.distributions.Normal(mean_pred, std_pred)
            old_lp    = dist_pred.log_prob(a_t).sum(-1)

        # f. PPO update with BC-KL regularization
        metrics = ppo_update(
            model, optimizer, s_t, a_t, old_lp, ret_t,
            bc_model=bc_ref_model,
            n_epochs=1, batch_size=256, clip_eps=0.05,
            kl_coef=0.5, device=device,
        )

        # Z-distance at final step (primary metric: port z = 0.190)
        # Use full state (index 2 = tcp.z) for these metrics
        final_z_dist = float(abs(ep_states_full[-1, 2] - TARGET_Z))
        # Approach rate: fraction of steps where z moved toward 0.190
        z_pos  = ep_states_full[:, 2]
        z_appr = float((np.diff(np.abs(z_pos - TARGET_Z)) < 0).mean())

        elapsed = time.time() - t0
        entry = {
            "iteration":       iteration,
            "steps":           T,
            "mean_reward":     mean_reward,
            "final_dist_m":    final_dist,
            "z_approach_rate": z_appr,
            "final_z_dist":    final_z_dist,
            **metrics,
            "elapsed_s": elapsed,
        }
        history.append(entry)
        json.dump(history, open(out_dir / "rl_history.json", "w"), indent=2)

        print(f"{iteration:>5}  {T:>6}  {z_appr:>6.3f}  {final_z_dist:>7.4f}  "
              f"{metrics['policy_loss']:>8.4f}  {metrics['value_loss']:>8.4f}  "
              f"{metrics['kl_bc']:>7.4f}  {elapsed:.0f}s")

        # g. Checkpoint every 5 iterations
        if iteration % 5 == 0:
            save_checkpoint(model, optimizer, iteration,
                            state_mean, state_std, port_pos, ckpt_dir)

    # Final save
    save_checkpoint(model, optimizer, args.rl_iter,
                    state_mean, state_std, port_pos, ckpt_dir)
    print(f"\nRL training done. Checkpoints: {ckpt_dir}")
    print(f"RL history:                    {out_dir / 'rl_history.json'}")
    print(f"Final weights:                  {MLP_DIR / 'weights.pt'}")


if __name__ == "__main__":
    main()
