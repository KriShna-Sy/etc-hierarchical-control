#!/usr/bin/env python3
"""
Three-way ETC comparison analyzer: ZOH hold vs Smart hold vs Adaptive-sigma
Smart hold, with repeated-trial mean +/- std for every condition.

Run from ~/etc_ws with plot_env active (source plot_env/bin/activate).

Expects bag/log naming:
  Repeated trials (n=3):  square_sigma{sigma}_{group}_t{trial}
                          square_sigma{sigma}_{group}_t{trial}_log.txt
  where group in {zoh, smart, adaptive_smart} and sigma in {0.1, 0.2}.

  Sanity check (n=1, optional, just printed for confirmation):
                          square_sigma0.1_smart_l3check
                          sigma0.1_smart_l3check_log.txt
"""
from rosbags.highlevel import AnyReader
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt

# ---------- Reference trajectory model (same square path as before) ----------
V_D = 0.2
OMEGA_TURN = 0.5
SIDE = 2.0
DT = 1.0 / 50.0
LOOP_PERIOD = 4 * (SIDE / V_D + (np.pi / 2) / OMEGA_TURN)


def _build_ideal_table():
    n_steps = int(np.ceil(LOOP_PERIOD / DT)) + 5
    xs = np.zeros(n_steps)
    ys = np.zeros(n_steps)
    x, y, theta = 0.0, 0.0, 0.0
    state = 'MOVE'
    dist_done, turn_done = 0.0, 0.0
    for i in range(n_steps):
        xs[i], ys[i] = x, y
        if state == 'MOVE':
            x += V_D * np.cos(theta) * DT
            y += V_D * np.sin(theta) * DT
            dist_done += V_D * DT
            if dist_done >= SIDE:
                dist_done = 0.0
                state = 'TURN'
        elif state == 'TURN':
            theta += OMEGA_TURN * DT
            turn_done += OMEGA_TURN * DT
            if turn_done >= np.pi / 2:
                turn_done = 0.0
                theta = round(theta / (np.pi / 2)) * (np.pi / 2)
                state = 'MOVE'
    return xs, ys


_IDEAL_X, _IDEAL_Y = _build_ideal_table()


def ideal_trajectory_at(t):
    t_wrapped = t % LOOP_PERIOD
    idx = int(t_wrapped / DT) % len(_IDEAL_X)
    return _IDEAL_X[idx], _IDEAL_Y[idx]


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


def find_best_offset(xs, ys, ts, max_offset=10.0, coarse_step=0.05, fine_step=0.005):
    t0 = ts[0]
    n = min(300, len(xs))
    idx = np.linspace(0, len(xs) - 1, n).astype(int)
    xs_s, ys_s, ts_s = xs[idx], ys[idx], ts[idx]
    rel_t = ts_s - t0

    def sse_for_offset(off):
        err = 0.0
        for x, y, t in zip(xs_s, ys_s, rel_t):
            xd, yd = ideal_trajectory_at(max(0.0, t + off))
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


def analyze_trial(bag_path, log_path):
    """Returns (ate_mean, events_per_sec, duration) for a single trial, or None if missing/short."""
    if not Path(bag_path).exists():
        print(f"  !! missing bag: {bag_path}")
        return None
    if not Path(log_path).exists():
        print(f"  !! missing log: {log_path}")
        return None
    try:
        xs, ys, ts = load_bag(bag_path)
        if len(xs) < 20:
            print(f"  !! too short: {bag_path} ({len(xs)} odom msgs)")
            return None
        ate_mean, ate_std, ate_max, offset = compute_ate(xs, ys, ts)
        duration = ts[-1] - ts[0]

        event_times, event_idx, total_ticks = parse_event_log(log_path)
        if len(event_times) > 1:
            run_duration = event_times[-1] - event_times[0]
            events_per_sec = len(event_times) / run_duration if run_duration > 0 else float('nan')
        else:
            events_per_sec = float('nan')

        return {
            'ate_mean': ate_mean, 'ate_std': ate_std, 'ate_max': ate_max,
            'offset': offset, 'duration': duration, 'events_per_sec': events_per_sec,
        }
    except Exception as e:
        print(f"  !! error processing {bag_path}: {e}")
        return None


# ---------- CONFIG: edit this if your naming differs ----------
SIGMAS = ['0.1', '0.2']
GROUPS = [
    ('zoh', 'ZOH hold'),
    ('smart', 'Smart hold'),
    ('adaptive_smart', 'Adaptive-sigma Smart hold'),
]
N_TRIALS = 5

# Optional single-run sanity check (adaptive_sigma:=false should match L2 smart-hold numbers)
SANITY_CHECK = ('square_sigma0.1_smart_l3check', 'sigma0.1_smart_l3check_log.txt')


def main():
    # --- Sanity check (optional) ---
    bag, log = SANITY_CHECK
    if Path(bag).exists():
        print(f"{'='*60}\nSANITY CHECK: {bag}\n{'='*60}")
        result = analyze_trial(bag, log)
        if result:
            print(f"ATE mean = {result['ate_mean']:.4f} m  "
                  f"(compare against your L2 smart-hold sigma=0.1 mean of ~0.0073m)")
            print(f"Events/sec = {result['events_per_sec']:.2f}")
        print()

    # --- Main grouped comparison ---
    all_results = {}  # (sigma, group_key) -> list of trial dicts
    for sigma in SIGMAS:
        for group_key, group_label in GROUPS:
            trials = []
            for t in range(1, N_TRIALS + 1):
                name = f"square_sigma{sigma}_{group_key}_t{t}"
                bag = name
                log = f"{name}_log.txt"
                print(f"Processing {name} ...")
                r = analyze_trial(bag, log)
                if r:
                    trials.append(r)
            all_results[(sigma, group_key)] = trials

    # --- Print summary table ---
    print(f"\n{'='*90}")
    print(f"{'Condition':<32}{'n':<4}{'ATE mean +/- std (m)':<28}{'Events/sec +/- std':<24}")
    print(f"{'='*90}")
    summary = {}
    for sigma in SIGMAS:
        for group_key, group_label in GROUPS:
            trials = all_results[(sigma, group_key)]
            label = f"sigma={sigma}, {group_label}"
            if len(trials) == 0:
                print(f"{label:<32}{0:<4}{'NO DATA':<28}{'NO DATA':<24}")
                continue
            ate_vals = np.array([t['ate_mean'] for t in trials])
            evt_vals = np.array([t['events_per_sec'] for t in trials])
            ate_str = f"{ate_vals.mean():.4f} +/- {ate_vals.std():.4f}"
            evt_str = f"{evt_vals.mean():.2f} +/- {evt_vals.std():.2f}"
            print(f"{label:<32}{len(trials):<4}{ate_str:<28}{evt_str:<24}")
            summary[(sigma, group_key)] = {
                'ate_mean': ate_vals.mean(), 'ate_std': ate_vals.std(),
                'evt_mean': evt_vals.mean(), 'evt_std': evt_vals.std(),
                'n': len(trials),
            }

    if not summary:
        print("\nNo data found yet -- run the experiments first, then re-run this script.")
        return

    # --- Figure: grouped bar chart with error bars, one panel per sigma ---
    fig, axes = plt.subplots(1, len(SIGMAS), figsize=(6 * len(SIGMAS), 5), sharey=False)
    if len(SIGMAS) == 1:
        axes = [axes]

    for ax, sigma in zip(axes, SIGMAS):
        labels = [g[1] for g in GROUPS if (sigma, g[0]) in summary]
        ate_means = [summary[(sigma, g[0])]['ate_mean'] for g in GROUPS if (sigma, g[0]) in summary]
        ate_stds = [summary[(sigma, g[0])]['ate_std'] for g in GROUPS if (sigma, g[0]) in summary]
        x = np.arange(len(labels))
        ax.bar(x, ate_means, yerr=ate_stds, capsize=6, color=['steelblue', 'darkorange', 'seagreen'][:len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('ATE mean (m)')
        ax.set_title(f'sigma = {sigma}')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('ATE Across Hold/Threshold Strategies (mean +/- std, n=3 trials)')
    fig.tight_layout()
    fig.savefig('fig_l3_three_way_comparison.png', dpi=150)
    print("\nSaved fig_l3_three_way_comparison.png")

    # --- Figure: events/sec companion chart ---
    fig2, axes2 = plt.subplots(1, len(SIGMAS), figsize=(6 * len(SIGMAS), 5), sharey=False)
    if len(SIGMAS) == 1:
        axes2 = [axes2]
    for ax, sigma in zip(axes2, SIGMAS):
        labels = [g[1] for g in GROUPS if (sigma, g[0]) in summary]
        evt_means = [summary[(sigma, g[0])]['evt_mean'] for g in GROUPS if (sigma, g[0]) in summary]
        evt_stds = [summary[(sigma, g[0])]['evt_std'] for g in GROUPS if (sigma, g[0]) in summary]
        x = np.arange(len(labels))
        ax.bar(x, evt_means, yerr=evt_stds, capsize=6, color=['steelblue', 'darkorange', 'seagreen'][:len(labels)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Events/sec')
        ax.set_title(f'sigma = {sigma}')
        ax.grid(axis='y', alpha=0.3)

    fig2.suptitle('Event Rate Across Hold/Threshold Strategies (mean +/- std, n=3 trials)')
    fig2.tight_layout()
    fig2.savefig('fig_l3_events_comparison.png', dpi=150)
    print("Saved fig_l3_events_comparison.png")

    plt.show()


if __name__ == '__main__':
    main()
