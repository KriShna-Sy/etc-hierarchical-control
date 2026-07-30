#!/usr/bin/env python3
"""
Circle trajectory: repeated-trials ZOH vs Smart-hold comparison with a real
Welch's t-test per (sigma) pair, matching the methodology already used for
the square trajectory in the paper (Section IV-B / Table III).

Expects bags/logs named:
    circle_sigma{sigma}_{mode}_t{trial}
    circle_sigma{sigma}_{mode}_t{trial}_log.txt

Run from ~/etc_ws with plot_env active.
"""
from rosbags.highlevel import AnyReader
from pathlib import Path
import numpy as np
import re
import math
import matplotlib.pyplot as plt

# ---------- CONFIG: circle trajectory parameters ----------
# NOTE: confirm these match your trajectory_generator_node's circle branch
# (radius / angular rate used when shape='circle'). Adjust if different.
V_D = 0.2
CIRCLE_OMEGA = 0.2   # rad/s, constant turning rate for the circle shape
DT = 1.0/50.0
LOOP_PERIOD = 2*np.pi / CIRCLE_OMEGA if CIRCLE_OMEGA > 0 else 60.0
RADIUS = V_D / CIRCLE_OMEGA if CIRCLE_OMEGA > 0 else 1.0

def _build_ideal_table():
    n_steps = int(np.ceil(LOOP_PERIOD/DT)) + 5
    xs = np.zeros(n_steps); ys = np.zeros(n_steps)
    theta = 0.0
    x, y = 0.0, 0.0
    for i in range(n_steps):
        xs[i], ys[i] = x, y
        x += V_D*np.cos(theta)*DT
        y += V_D*np.sin(theta)*DT
        theta += CIRCLE_OMEGA*DT
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
    event_times, total_ticks = [], []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                event_times.append(float(m.group(1)))
                total_ticks.append(int(m.group(3)))
    return np.array(event_times), np.array(total_ticks)

def analyze_trial(bag_path, log_path, run_duration_fallback):
    """events_per_sec computed over TOTAL RUN DURATION (from odom timestamps),
    not event-to-event span -- this is the bug that was fixed for the
    single-trial circle pilot; keep that fix here."""
    try:
        xs, ys, ts = load_bag(bag_path)
        if len(xs) < 10:
            print(f"  !! {bag_path}: too few odom messages ({len(xs)}), skipping")
            return None
        ate_mean, ate_std, ate_max, offset = compute_ate(xs, ys, ts)
        run_duration = ts[-1] - ts[0]

        event_times, total_ticks = parse_event_log(log_path)
        n_events = len(event_times)
        events_per_sec = n_events / run_duration if run_duration > 0 else float('nan')
        n_ticks = total_ticks[-1] if len(total_ticks) > 0 else float('nan')
        pct_fired = 100.0 * n_events / n_ticks if n_ticks and n_ticks > 0 else 0.0

        return {'ate_mean': ate_mean, 'events_per_sec': events_per_sec,
                'pct_fired': pct_fired, 'n_events': n_events, 'duration': run_duration}
    except Exception as e:
        print(f"  !! Failed to analyze {bag_path}: {e}")
        return None

def welch_t_test(a, b):
    """Two-sample Welch's t-test from raw sample arrays a, b.
    Returns (t, df, p_two_sided). Uses the same Welch-Satterthwaite
    approximation described in the paper; p-value via a numerical
    approximation to the Student-t CDF (no scipy dependency required)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va/na + vb/nb)
    if se == 0:
        return float('nan'), float('nan'), float('nan')
    t = (ma - mb) / se
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))

    # two-sided p-value via Student-t survival function (Hill's approximation)
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
    except ImportError:
        # Fallback: normal approximation (fine for df > ~5, which we have)
        from math import erf
        z = abs(t)
        p = 2 * (1 - 0.5*(1 + erf(z/math.sqrt(2))))
    return t, df, p

# ---------- ANALYZE ALL TRIALS ----------
SIGMAS = [0.1, 0.2]
MODES = ['zoh', 'smart']
TRIALS = [1, 2, 3, 4, 5]

raw = {}  # (sigma, mode) -> list of per-trial result dicts
for sigma in SIGMAS:
    for mode in MODES:
        key = (sigma, mode)
        raw[key] = []
        for trial in TRIALS:
            name = f"circle_sigma{sigma}_{mode}_t{trial}"
            print(f"Analyzing {name} ...")
            r = analyze_trial(name, f"{name}_log.txt", run_duration_fallback=49.0)
            if r is not None:
                raw[key].append(r)

# ---------- SUMMARY TABLE ----------
print(f"\n{'='*90}")
print(f"{'Condition':<20}{'n':<4}{'ATE mean +/- std (m)':<28}{'Events/sec mean +/- std':<28}")
print(f"{'='*90}")
summary = {}
for (sigma, mode), trials in raw.items():
    n = len(trials)
    if n == 0:
        print(f"sigma={sigma} {mode:<8} 0    NO VALID TRIALS")
        continue
    ate_vals = np.array([t['ate_mean'] for t in trials])
    evt_vals = np.array([t['events_per_sec'] for t in trials])
    summary[(sigma, mode)] = {'ate_vals': ate_vals, 'evt_vals': evt_vals, 'n': n}
    label = f"sigma={sigma} {mode}"
    print(f"{label:<20}{n:<4}{ate_vals.mean():.4f} +/- {ate_vals.std(ddof=1):.4f}          "
          f"{evt_vals.mean():.2f} +/- {evt_vals.std(ddof=1):.2f}")

# ---------- WELCH'S T-TEST: ZOH vs SMART, PER SIGMA, EACH METRIC ----------
print(f"\n{'='*90}")
print("Welch's t-test: ZOH vs Smart hold (circle trajectory)")
print(f"{'='*90}")
for sigma in SIGMAS:
    if (sigma, 'zoh') in summary and (sigma, 'smart') in summary:
        z, s = summary[(sigma, 'zoh')], summary[(sigma, 'smart')]
        for metric in ['ate_vals', 'evt_vals']:
            label = 'ATE' if metric == 'ate_vals' else 'events/sec'
            t, df, p = welch_t_test(z[metric], s[metric])
            sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
            print(f"sigma={sigma} {label}: ZOH={z[metric].mean():.4f}+/-{z[metric].std(ddof=1):.4f}  "
                  f"Smart={s[metric].mean():.4f}+/-{s[metric].std(ddof=1):.4f}  "
                  f"t({df:.1f})={t:.2f}  p={p:.3f}  [{sig}]")

# ---------- FIGURE ----------
if summary:
    fig, axes = plt.subplots(1, 2, figsize=(11,5))
    for ax, sigma in zip(axes, SIGMAS):
        z = summary.get((sigma, 'zoh'))
        s = summary.get((sigma, 'smart'))
        if z is None or s is None:
            continue
        means = [z['ate_vals'].mean(), s['ate_vals'].mean()]
        stds = [z['ate_vals'].std(ddof=1), s['ate_vals'].std(ddof=1)]
        ax.bar(['ZOH', 'Smart hold'], means, yerr=stds, capsize=5, color=['gray', 'seagreen'])
        ax.set_title(f'sigma={sigma}')
        ax.set_ylabel('ATE mean (m)')
    fig.suptitle('Circle trajectory: ZOH vs Smart Hold ATE (n=5 trials)')
    fig.tight_layout()
    plt.savefig('fig_circle_zoh_vs_smart_errorbars.png', dpi=150)
    print("\nSaved fig_circle_zoh_vs_smart_errorbars.png")

plt.show()
