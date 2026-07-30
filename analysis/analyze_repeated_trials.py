#!/usr/bin/env python3
"""
Repeated-trials analysis: mean +/- std ATE and events/sec across 3 trials
per (sigma, hold_mode) condition, plus a grouped bar chart with error bars.

Expects bags/logs named exactly as produced by the repeated-trials loop:
    square_sigma{sigma}_{mode}_t{trial}          (bag folder)
    square_sigma{sigma}_{mode}_t{trial}_log.txt  (event log)
e.g. square_sigma0.1_zoh_t1, square_sigma0.1_zoh_t1_log.txt

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

def analyze_trial(bag_path, log_path):
    """Returns (ate_mean, events_per_sec, pct_fired, t_min_ms) for one trial,
    or None if the bag/log couldn't be read (missing/too short)."""
    try:
        xs, ys, ts = load_bag(bag_path)
        if len(xs) < 10:
            print(f"  !! {bag_path}: too few odom messages ({len(xs)}), skipping")
            return None
        ate_mean, ate_std, ate_max, offset = compute_ate(xs, ys, ts)

        event_times, event_idx, total_ticks = parse_event_log(log_path)
        if len(event_times) > 1:
            run_duration = event_times[-1] - event_times[0]
            events_per_sec = len(event_times) / run_duration
            n_ticks = total_ticks[-1]
            pct_fired = 100.0 * len(event_times) / n_ticks if n_ticks > 0 else float('nan')
            t_min = np.diff(event_times).min() * 1000
        else:
            events_per_sec = pct_fired = t_min = float('nan')

        return {
            'ate_mean': ate_mean, 'ate_std': ate_std, 'ate_max': ate_max,
            'events_per_sec': events_per_sec, 'pct_fired': pct_fired, 't_min_ms': t_min,
            'duration': ts[-1] - ts[0]
        }
    except Exception as e:
        print(f"  !! Failed to analyze {bag_path}: {e}")
        return None

# ---------- BUILD CONDITION LIST FROM NAMING CONVENTION ----------
SIGMAS = [0.1, 0.2]
MODES = ['zoh', 'smart']
TRIALS = [1, 2, 3]

grouped_results = {}  # (sigma, mode) -> list of per-trial dicts

for sigma in SIGMAS:
    for mode in MODES:
        key = (sigma, mode)
        grouped_results[key] = []
        for trial in TRIALS:
            name = f"square_sigma{sigma}_{mode}_t{trial}"
            bag = name
            log = f"{name}_log.txt"
            print(f"Analyzing {name} ...")
            r = analyze_trial(bag, log)
            if r is not None:
                grouped_results[key].append(r)

# ---------- PRINT MEAN +/- STD PER CONDITION ----------
print(f"\n{'='*90}")
print(f"{'Condition':<20}{'n':<4}{'ATE mean +/- std (m)':<28}{'Events/sec mean +/- std':<28}")
print(f"{'='*90}")

summary = {}
for (sigma, mode), trials in grouped_results.items():
    n = len(trials)
    if n == 0:
        print(f"sigma={sigma} {mode:<8} 0    NO VALID TRIALS")
        continue
    ate_vals = np.array([t['ate_mean'] for t in trials])
    evt_vals = np.array([t['events_per_sec'] for t in trials])
    ate_mu, ate_sd = ate_vals.mean(), ate_vals.std()
    evt_mu, evt_sd = evt_vals.mean(), evt_vals.std()
    summary[(sigma, mode)] = {
        'ate_mu': ate_mu, 'ate_sd': ate_sd,
        'evt_mu': evt_mu, 'evt_sd': evt_sd, 'n': n
    }
    label = f"sigma={sigma} {mode}"
    print(f"{label:<20}{n:<4}{ate_mu:.4f} +/- {ate_sd:.4f}          {evt_mu:.2f} +/- {evt_sd:.2f}")

# ---------- ZOH vs SMART COMPARISON WITH VARIANCE ----------
print(f"\n{'='*90}")
print("ZOH vs SMART HOLD (mean +/- std across trials):")
print(f"{'='*90}")
for sigma in SIGMAS:
    if (sigma, 'zoh') in summary and (sigma, 'smart') in summary:
        z = summary[(sigma, 'zoh')]
        s = summary[(sigma, 'smart')]
        pct_change = 100.0 * (s['ate_mu'] - z['ate_mu']) / z['ate_mu']
        print(f"sigma={sigma}:")
        print(f"  ZOH   ATE = {z['ate_mu']:.4f} +/- {z['ate_sd']:.4f} m  (n={z['n']}, events/sec={z['evt_mu']:.2f}+/-{z['evt_sd']:.2f})")
        print(f"  Smart ATE = {s['ate_mu']:.4f} +/- {s['ate_sd']:.4f} m  (n={s['n']}, events/sec={s['evt_mu']:.2f}+/-{s['evt_sd']:.2f})")
        print(f"  Change: {pct_change:+.1f}%")

# ---------- FIGURE: grouped bar chart with error bars ----------
if summary:
    fig, ax = plt.subplots(figsize=(7,5))
    x = np.arange(len(SIGMAS))
    width = 0.35

    zoh_mu = [summary.get((s,'zoh'), {}).get('ate_mu', np.nan) for s in SIGMAS]
    zoh_sd = [summary.get((s,'zoh'), {}).get('ate_sd', 0) for s in SIGMAS]
    smart_mu = [summary.get((s,'smart'), {}).get('ate_mu', np.nan) for s in SIGMAS]
    smart_sd = [summary.get((s,'smart'), {}).get('ate_sd', 0) for s in SIGMAS]

    ax.bar(x - width/2, zoh_mu, width, yerr=zoh_sd, capsize=5, label='ZOH hold (L1)', color='gray')
    ax.bar(x + width/2, smart_mu, width, yerr=smart_sd, capsize=5, label='Smart hold (L2)', color='seagreen')
    ax.set_xticks(x); ax.set_xticklabels([f"sigma={s}" for s in SIGMAS])
    ax.set_ylabel('ATE mean (m)')
    ax.set_title(f'ZOH vs Smart Hold: ATE (mean +/- std, n={TRIALS[-1]} trials)')
    ax.legend()
    fig.tight_layout()
    plt.savefig('fig_zoh_vs_smart_hold_errorbars.png', dpi=150)
    print("\nSaved fig_zoh_vs_smart_hold_errorbars.png")

plt.show()
