#!/usr/bin/env python3
"""
Full ETC experiment analysis: ATE (with auto phase-alignment), events/sec,
no-Zeno check, comparison charts.

Extended for L2: supports comparing hold_mode='zoh' vs hold_mode='smart'
at matched sigma. Event logs are unaffected by hold_mode (the trigger
decision itself doesn't change — only what the actuator does between
events), so events/sec, %fired, and T_min are computed exactly as before.

Run from ~/etc_ws with plot_env active (source plot_env/bin/activate).
"""
from rosbags.highlevel import AnyReader
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt

# ---------- CONFIG: matches your trajectory_generator_node ----------
V_D = 0.2
OMEGA_TURN = 0.5
SIDE = 2.0
DT = 1.0/50.0

LOOP_PERIOD = 4*(SIDE/V_D + (np.pi/2)/OMEGA_TURN)

def _build_ideal_table():
    n_steps = int(np.ceil(LOOP_PERIOD/DT)) + 5
    xs = np.zeros(n_steps); ys = np.zeros(n_steps)
    x, y, theta = 0.0, 0.0, 0.0
    state = 'MOVE'
    dist_done, turn_done = 0.0, 0.0
    for i in range(n_steps):
        xs[i], ys[i] = x, y
        if state == 'MOVE':
            x += V_D*np.cos(theta)*DT
            y += V_D*np.sin(theta)*DT
            dist_done += V_D*DT
            if dist_done >= SIDE:
                dist_done = 0.0
                state = 'TURN'
        elif state == 'TURN':
            theta += OMEGA_TURN*DT
            turn_done += OMEGA_TURN*DT
            if turn_done >= np.pi/2:
                turn_done = 0.0
                theta = round(theta/(np.pi/2))*(np.pi/2)
                state = 'MOVE'
    return xs, ys

_IDEAL_X, _IDEAL_Y = _build_ideal_table()

def ideal_trajectory_at(t):
    t_wrapped = t % LOOP_PERIOD
    idx = int(t_wrapped / DT) % len(_IDEAL_X)
    return _IDEAL_X[idx], _IDEAL_Y[idx]

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
    idx = np.linspace(0, len(xs)-1, n).astype(int)
    xs_s, ys_s, ts_s = xs[idx], ys[idx], ts[idx]
    rel_t = ts_s - t0

    def sse_for_offset(off):
        err = 0.0
        for x, y, t in zip(xs_s, ys_s, rel_t):
            xd, yd = ideal_trajectory_at(t + off)
            err += (x - xd)**2 + (y - yd)**2
        return err

    coarse_offsets = np.arange(0.0, max_offset + coarse_step, coarse_step)
    coarse_errs = [sse_for_offset(o) for o in coarse_offsets]
    best_coarse = coarse_offsets[int(np.argmin(coarse_errs))]

    fine_lo = max(0.0, best_coarse - coarse_step)
    fine_hi = best_coarse + coarse_step
    fine_offsets = np.arange(fine_lo, fine_hi + fine_step, fine_step)
    fine_errs = [sse_for_offset(o) for o in fine_offsets]
    best_offset = fine_offsets[int(np.argmin(fine_errs))]
    return best_offset

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
    if len(event_times) > 1:
        run_duration = event_times[-1] - event_times[0]
        events_per_sec = len(event_times) / run_duration
        n_ticks = total_ticks[-1]
        pct_fired = 100.0 * len(event_times) / n_ticks if n_ticks > 0 else float('nan')
        inter_event = np.diff(event_times)
        t_min = inter_event.min()
        t_mean = inter_event.mean()
        zeno_flag = "OK (no-Zeno)" if t_min > 1e-4 else "WARNING: near-zero interval detected"
    else:
        events_per_sec = n_ticks = pct_fired = t_min = t_mean = float('nan')
        zeno_flag = "N/A"

    print(f"Events fired: {len(event_times)} / {n_ticks} ticks ({pct_fired:.1f}%)")
    print(f"Events/sec: {events_per_sec:.2f}")
    print(f"Min inter-event interval T_min: {t_min*1000:.2f} ms")
    print(f"Mean inter-event interval: {t_mean*1000:.2f} ms")
    print(f"Zeno check: {zeno_flag}")

    return {
        'name': name, 'sigma': sigma, 'mode': mode,
        'xs': xs, 'ys': ys,
        'ate_mean': ate_mean, 'ate_std': ate_std, 'ate_max': ate_max,
        'offset': offset,
        'events_per_sec': events_per_sec, 'pct_fired': pct_fired,
        't_min_ms': t_min*1000
    }

# ---------- CONDITIONS ----------
# Add 'sigma' (float, use 0.0 for continuous) and 'mode' ('zoh' or 'smart')
# to every entry so results can be grouped/compared automatically.
# Fill in the bag/log names for your new smart-hold runs below once you've
# recorded them (see run_experiment_launch.py hold_mode:=smart).
conditions = [
    {"name": "Continuous (sigma=0.0)", "bag": "square_sigma0_final2",   "log": "sigma0_final2_log.txt",   "sigma": 0.0, "mode": "zoh"},
    {"name": "ZOH (sigma=0.1)",        "bag": "square_sigma0.1_final2", "log": "sigma0.1_final2_log.txt", "sigma": 0.1, "mode": "zoh"},
    {"name": "ZOH (sigma=0.2)",        "bag": "square_sigma0.2_final2", "log": "sigma0.2_final2_log.txt", "sigma": 0.2, "mode": "zoh"},
    # ---- L2 additions: fill these in after running hold_mode:=smart ----
    {"name": "Smart (sigma=0.1)", "bag": "square_sigma0.1_smart", "log": "sigma0.1_smart_log.txt", "sigma": 0.1, "mode": "smart"},
    {"name": "Smart (sigma=0.2)", "bag": "square_sigma0.2_smart", "log": "sigma0.2_smart_log.txt", "sigma": 0.2, "mode": "smart"},
]

results = []
for cond in conditions:
    try:
        results.append(analyze_condition(cond["name"], cond["bag"], cond["log"],
                                          sigma=cond["sigma"], mode=cond["mode"]))
    except Exception as e:
        print(f"\n!! Failed to analyze {cond['name']}: {e}")

# ---------- SUMMARY TABLE ----------
print(f"\n{'='*95}")
print(f"{'Condition':<25}{'sigma':<8}{'mode':<8}{'ATE mean (m)':<15}{'Offset (s)':<12}{'Events/sec':<13}{'% ticks fired':<15}")
print(f"{'='*95}")
for r in results:
    print(f"{r['name']:<25}{r['sigma']:<8}{r['mode']:<8}{r['ate_mean']:<15.4f}{r['offset']:<12.3f}{r['events_per_sec']:<13.2f}{r['pct_fired']:<15.1f}")

# ---------- FIGURE 1: Trajectory overlay ----------
plt.figure(figsize=(7,7))
colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))
for r, c in zip(results, colors):
    plt.plot(r['xs'], r['ys'], '-', linewidth=1, color=c, label=r['name'], alpha=0.8)
plt.xlabel('X (m)'); plt.ylabel('Y (m)')
plt.title('Trajectory Comparison Across Conditions')
plt.axis('equal'); plt.grid(True, alpha=0.3); plt.legend(fontsize=8)
plt.savefig('fig_trajectory_comparison.png', dpi=150)
print("\nSaved fig_trajectory_comparison.png")

# ---------- FIGURE 2: Bandwidth-accuracy tradeoff bar chart (all conditions) ----------
fig, ax1 = plt.subplots(figsize=(9,5))
names = [r['name'] for r in results]
ate_vals = [r['ate_mean'] for r in results]
evt_vals = [r['events_per_sec'] for r in results]

x = np.arange(len(names))
width = 0.35
ax1.bar(x - width/2, ate_vals, width, label='ATE mean (m)', color='steelblue')
ax1.set_ylabel('ATE mean (m)', color='steelblue')
ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha='right')

ax2 = ax1.twinx()
ax2.bar(x + width/2, evt_vals, width, label='Events/sec', color='darkorange')
ax2.set_ylabel('Events/sec', color='darkorange')

plt.title('Bandwidth (events/sec) vs Accuracy (ATE) Tradeoff')
fig.tight_layout()
plt.savefig('fig_bandwidth_accuracy_tradeoff.png', dpi=150)
print("Saved fig_bandwidth_accuracy_tradeoff.png")

# ---------- FIGURE 3: ZOH vs Smart, grouped by sigma (only if both modes present) ----------
sigmas_with_both = sorted(set(
    r['sigma'] for r in results
    if any(r2['sigma'] == r['sigma'] and r2['mode'] == 'smart' for r2 in results)
    and any(r2['sigma'] == r['sigma'] and r2['mode'] == 'zoh' for r2 in results)
))

if sigmas_with_both:
    zoh_ate = [next(r['ate_mean'] for r in results if r['sigma']==s and r['mode']=='zoh') for s in sigmas_with_both]
    smart_ate = [next(r['ate_mean'] for r in results if r['sigma']==s and r['mode']=='smart') for s in sigmas_with_both]

    x = np.arange(len(sigmas_with_both))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7,5))
    ax.bar(x - width/2, zoh_ate, width, label='ZOH hold (L1)', color='gray')
    ax.bar(x + width/2, smart_ate, width, label='Smart hold (L2)', color='seagreen')
    ax.set_xticks(x); ax.set_xticklabels([f"sigma={s}" for s in sigmas_with_both])
    ax.set_ylabel('ATE mean (m)')
    ax.set_title('ZOH vs Smart Hold: ATE at Matched Sigma\n(events/sec should be ~equal within each pair)')
    ax.legend()
    fig.tight_layout()
    plt.savefig('fig_zoh_vs_smart_hold.png', dpi=150)
    print("Saved fig_zoh_vs_smart_hold.png")

    print(f"\n{'='*60}")
    print("ZOH vs SMART HOLD comparison (matched sigma):")
    print(f"{'='*60}")
    for s, za, sa in zip(sigmas_with_both, zoh_ate, smart_ate):
        pct_change = 100.0 * (sa - za) / za
        print(f"sigma={s}: ZOH ATE={za:.4f}m  Smart ATE={sa:.4f}m  ({pct_change:+.1f}%)")
else:
    print("\n(No matched ZOH/Smart pairs found yet — add smart-hold conditions above once recorded.)")

plt.show()
