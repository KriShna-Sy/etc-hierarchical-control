#!/usr/bin/env python3
"""
Full ETC experiment analysis: ATE, events/sec, no-Zeno check, comparison chart.
Run from ~/etc_ws with the plot_env venv active (source plot_env/bin/activate).
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

def ideal_trajectory_at(t):
    """Reconstruct the ideal square trajectory position at time t (seconds since start)."""
    x, y, theta = 0.0, 0.0, 0.0
    state = 'MOVE'
    dist_done, turn_done = 0.0, 0.0
    steps = int(t / DT)
    for _ in range(steps):
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
    return x, y

def load_bag(bag_path):
    xs, ys, ts = [], [], []
    with AnyReader([Path(bag_path)]) as reader:
        connections = [c for c in reader.connections if c.topic == '/diff_drive_controller/odom']
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            xs.append(msg.pose.pose.position.x)
            ys.append(msg.pose.pose.position.y)
            ts.append(timestamp / 1e9)
    return np.array(xs), np.array(ys), np.array(ts)

def compute_ate(xs, ys, ts):
    t0 = ts[0]
    errors = []
    for x, y, t in zip(xs, ys, ts):
        xd, yd = ideal_trajectory_at(t - t0)
        errors.append(np.hypot(x - xd, y - yd))
    errors = np.array(errors)
    return errors.mean(), errors.std(), errors.max()

def parse_event_log(log_path):
    """Extract event timestamps from ros2 log lines like:
    [INFO] [1783083574.021477137] [event_trigger_node]: EVENT 4/4 e=1.974 thr=0.000
    """
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

def analyze_condition(name, bag_path, log_path):
    print(f"\n{'='*50}\n{name}\n{'='*50}")

    xs, ys, ts = load_bag(bag_path)
    duration = ts[-1] - ts[0]
    print(f"Odom messages: {len(xs)}  |  Duration: {duration:.1f}s")

    ate_mean, ate_std, ate_max = compute_ate(xs, ys, ts)
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
        'name': name, 'xs': xs, 'ys': ys,
        'ate_mean': ate_mean, 'ate_std': ate_std, 'ate_max': ate_max,
        'events_per_sec': events_per_sec, 'pct_fired': pct_fired,
        't_min_ms': t_min*1000
    }

# ---------- RUN ALL THREE CONDITIONS ----------
conditions = [
    ("Continuous (sigma=0.0)", "square_sigma0_final2",   "sigma0_final2_log.txt"),
    ("ETC (sigma=0.1)",        "square_sigma0.1_final2", "sigma0.1_final2_log.txt"),
    ("ETC (sigma=0.2)",        "square_sigma0.2_final2", "sigma0.2_final2_log.txt"),
]

results = []
for name, bag, log in conditions:
    try:
        results.append(analyze_condition(name, bag, log))
    except Exception as e:
        print(f"\n!! Failed to analyze {name}: {e}")

# ---------- SUMMARY TABLE ----------
print(f"\n{'='*70}")
print(f"{'Condition':<25}{'ATE mean (m)':<15}{'Events/sec':<15}{'% ticks fired':<15}")
print(f"{'='*70}")
for r in results:
    print(f"{r['name']:<25}{r['ate_mean']:<15.4f}{r['events_per_sec']:<15.2f}{r['pct_fired']:<15.1f}")

# ---------- FIGURE 1: Trajectory overlay ----------
plt.figure(figsize=(7,7))
colors = ['gray', 'steelblue', 'darkorange']
for r, c in zip(results, colors):
    plt.plot(r['xs'], r['ys'], '-', linewidth=1, color=c, label=r['name'], alpha=0.8)
plt.xlabel('X (m)'); plt.ylabel('Y (m)')
plt.title('Trajectory Comparison Across Conditions')
plt.axis('equal'); plt.grid(True, alpha=0.3); plt.legend()
plt.savefig('fig_trajectory_comparison.png', dpi=150)
print("\nSaved fig_trajectory_comparison.png")

# ---------- FIGURE 2: Bandwidth-accuracy tradeoff bar chart ----------
fig, ax1 = plt.subplots(figsize=(8,5))
names = [r['name'] for r in results]
ate_vals = [r['ate_mean'] for r in results]
evt_vals = [r['events_per_sec'] for r in results]

x = np.arange(len(names))
width = 0.35
ax1.bar(x - width/2, ate_vals, width, label='ATE mean (m)', color='steelblue')
ax1.set_ylabel('ATE mean (m)', color='steelblue')
ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15)

ax2 = ax1.twinx()
ax2.bar(x + width/2, evt_vals, width, label='Events/sec', color='darkorange')
ax2.set_ylabel('Events/sec', color='darkorange')

plt.title('Bandwidth (events/sec) vs Accuracy (ATE) Tradeoff')
fig.tight_layout()
plt.savefig('fig_bandwidth_accuracy_tradeoff.png', dpi=150)
print("Saved fig_bandwidth_accuracy_tradeoff.png")

plt.show()
