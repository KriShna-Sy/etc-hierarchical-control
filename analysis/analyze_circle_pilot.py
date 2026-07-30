#!/usr/bin/env python3
"""
Analyzer for the CIRCLE trajectory pilot (sigma-sweep + single-trial
hold-strategy check). Same logic as analyze_experiment.py, except the
idealized reference trajectory model matches trajectory_generator_node.py's
shape='circle' math exactly (constant curvature, no corners) instead of
the square's piecewise MOVE/TURN state machine.

IMPORTANT: if you change v_d or radius in trajectory_generator_node.py's
_step_circle(), update V_D / RADIUS below to match, or the ATE alignment
will silently compare against the wrong idealized path.

Run from ~/etc_ws with plot_env active (source plot_env/bin/activate).
"""
from rosbags.highlevel import AnyReader
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt

# ---------- CONFIG: must match trajectory_generator_node.py shape='circle' ----------
V_D = 0.2
RADIUS = 1.0
OMEGA_D = V_D / RADIUS          # constant, rad/s
LOOP_PERIOD = 2 * np.pi / OMEGA_D  # time for one full lap
DT = 1.0/50.0


def ideal_trajectory_at(t):
    """Closed-form -- no lookup table needed since the circle has no
    discrete states (unlike the square's MOVE/TURN machine)."""
    t_wrapped = t % LOOP_PERIOD
    theta = OMEGA_D * t_wrapped
    x = RADIUS * np.sin(theta)
    y = RADIUS * (1.0 - np.cos(theta))
    return x, y


def load_bag(bag_path, topic='/diff_drive_controller/odom'):
    xs, ys, ts = [], [], []
    with AnyReader([Path(bag_path)]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            xs.append(msg.pose.pose.position.x)
            ys.append(msg.pose.pose.position.y)
            ts.append(timestamp / 1e9)
    return np.array(xs), np.array(ys), np.array(ts)


def find_best_offset(xs, ys, ts, max_offset=10.0, coarse_step=0.05, fine_step=0.005):
    t0 = ts[0]
    n = min(300, len(xs))
    idx = np.linspace(0, len(xs) - 1, n).astype(int)
    xs_s, ys_s, ts_s = xs[idx], ys[idx], ts[idx]
    rel_t = ts_s - t0

    def sse_for_offset(off):
        err = 0.0
        for x, y, t in zip(xs_s, ys_s, rel_t):
            xd, yd = ideal_trajectory_at(t + off)
            err += (x - xd) ** 2 + (y - yd) ** 2
        return err

    coarse_offsets = np.arange(0.0, max_offset + coarse_step, coarse_step)
    coarse_errs = [sse_for_offset(o) for o in coarse_offsets]
    best_coarse = coarse_offsets[int(np.argmin(coarse_errs))]

    fine_lo = max(0.0, best_coarse - coarse_step)
    fine_hi = best_coarse + coarse_step
    fine_offsets = np.arange(fine_lo, fine_hi + fine_step, fine_step)
    fine_errs = [sse_for_offset(o) for o in fine_offsets]
    return fine_offsets[int(np.argmin(fine_errs))]


def compute_ate(xs, ys, ts):
    t0 = ts[0]
    offset = find_best_offset(xs, ys, ts)
    errors = []
    for x, y, t in zip(xs, ys, ts):
        t_adj = (t - t0) + offset
        xd, yd = ideal_trajectory_at(t_adj)
        errors.append(np.hypot(x - xd, y - yd))
    errors = np.array(errors)
    return errors.mean(), errors.std(), errors.max(), offset


def parse_event_log(log_path):
    pattern = re.compile(r'\[(\d+\.\d+)\].*EVENT (\d+)/(\d+)')
    event_times, event_idx, total_ticks = [], [], []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                event_times.append(float(m.group(1)))
                event_idx.append(int(m.group(2)))
                total_ticks.append(int(m.group(3)))
    return np.array(event_times), np.array(event_idx), np.array(total_ticks)


def analyze_condition(name, bag_path, log_path, sigma=None, mode='zoh'):
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    xs, ys, ts = load_bag(bag_path)
    duration = ts[-1] - ts[0]
    print(f"Odom messages: {len(xs)}  |  Duration: {duration:.1f}s")

    ate_mean, ate_std, ate_max, offset = compute_ate(xs, ys, ts)
    print(f"Calibrated phase offset: {offset:.3f}s")
    print(f"ATE:  mean={ate_mean:.4f}m  std={ate_std:.4f}m  max={ate_max:.4f}m")

    event_times, event_idx, total_ticks = parse_event_log(log_path)
    if len(event_times) >= 1:
        # IMPORTANT: normalize by the FULL run duration (from odom
        # timestamps), not by (event_times[-1] - event_times[0]). The
        # latter breaks badly whenever events are sparse and clustered
        # early in the run (e.g. smart/adaptive hold on sustained
        # curvature): a handful of events in a fraction of a second
        # produces a wildly inflated rate that doesn't reflect the true
        # communication cost averaged over the whole experiment.
        events_per_sec = len(event_times) / duration
        n_ticks = total_ticks[-1] if len(total_ticks) else 0
        pct_fired = 100.0 * len(event_times) / n_ticks if n_ticks > 0 else float('nan')
        t_min = np.diff(event_times).min() if len(event_times) > 1 else float('nan')
        if len(event_times) > 1 and t_min > 1e-4:
            zeno_flag = "OK (no-Zeno)"
        elif len(event_times) > 1:
            zeno_flag = "WARNING: near-zero interval detected"
        else:
            zeno_flag = "N/A (only 1 event fired)"
    else:
        # Zero events is a MEANINGFUL result here (the controller never
        # needed to fire again after startup), not missing data -- report
        # 0.0, not nan, or this looks like a failure when it may not be one.
        events_per_sec = 0.0
        n_ticks = total_ticks[-1] if len(total_ticks) else 0
        pct_fired = 0.0
        t_min = float('nan')
        zeno_flag = "N/A (zero events fired -- verify this is genuine, not a dead node)"

    print(f"Events/sec: {events_per_sec:.2f}  |  % ticks fired: {pct_fired:.1f}  |  T_min: {t_min*1000:.2f} ms  |  {zeno_flag}")

    return {'name': name, 'sigma': sigma, 'mode': mode, 'xs': xs, 'ys': ys,
             'ate_mean': ate_mean, 'ate_std': ate_std, 'events_per_sec': events_per_sec,
             'pct_fired': pct_fired}


# ---------- CONDITIONS: pilot sweep + single-trial hold-strategy check ----------
# Fill in bag/log names once you've run these. Suggested naming to match
# your existing convention: circle_sigma{sigma}_{mode}
conditions = [
    {"name": "Circle Continuous (sigma=0.0)", "bag": "circle_sigma0_zoh",   "log": "circle_sigma0_zoh_log.txt",   "sigma": 0.0, "mode": "zoh"},
    {"name": "Circle ETC (sigma=0.1)",        "bag": "circle_sigma0.1_zoh", "log": "circle_sigma0.1_zoh_log.txt", "sigma": 0.1, "mode": "zoh"},
    {"name": "Circle ETC (sigma=0.2)",        "bag": "circle_sigma0.2_zoh", "log": "circle_sigma0.2_zoh_log.txt", "sigma": 0.2, "mode": "zoh"},
    {"name": "Circle Smart hold (sigma=0.1)", "bag": "circle_sigma0.1_smart", "log": "circle_sigma0.1_smart_log.txt", "sigma": 0.1, "mode": "smart"},
    {"name": "Circle Smart hold (sigma=0.2)", "bag": "circle_sigma0.2_smart", "log": "circle_sigma0.2_smart_log.txt", "sigma": 0.2, "mode": "smart"},
    {"name": "Circle Adaptive-smart (sigma=0.1)", "bag": "circle_sigma0.1_adaptive_smart", "log": "circle_sigma0.1_adaptive_smart_log.txt", "sigma": 0.1, "mode": "adaptive_smart"},
    {"name": "Circle Adaptive-smart (sigma=0.2)", "bag": "circle_sigma0.2_adaptive_smart", "log": "circle_sigma0.2_adaptive_smart_log.txt", "sigma": 0.2, "mode": "adaptive_smart"},
]

results = []
for cond in conditions:
    try:
        results.append(analyze_condition(cond["name"], cond["bag"], cond["log"], sigma=cond["sigma"], mode=cond["mode"]))
    except Exception as e:
        print(f"\n!! Skipping {cond['name']}: {e}")

print(f"\n{'='*90}")
print(f"{'Condition':<32}{'sigma':<8}{'mode':<16}{'ATE mean (m)':<15}{'Events/sec':<12}")
print(f"{'='*90}")
for r in results:
    print(f"{r['name']:<32}{r['sigma']:<8}{r['mode']:<16}{r['ate_mean']:<15.4f}{r['events_per_sec']:<12.2f}")

if results:
    plt.figure(figsize=(6, 6))
    for r in results:
        plt.plot(r['xs'], r['ys'], linewidth=1, alpha=0.7, label=r['name'])
    plt.axis('equal'); plt.grid(alpha=0.3); plt.legend(fontsize=7)
    plt.title('Circle Trajectory: Pilot Sweep + Hold-Strategy Check')
    plt.savefig('fig_circle_trajectory_comparison.png', dpi=150)
    print("\nSaved fig_circle_trajectory_comparison.png")
plt.show()
