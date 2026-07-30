# Event-Triggered Trajectory Tracking for a Differential-Drive Robot

Simulation study of event-triggered control (ETC) for a differential-drive
robot in Gazebo/ROS 2: a threshold-based trigger, a "smart hold" actuation
strategy that tracks a live feedforward reference between events instead of
a naive zero-order hold (ZOH), and a curvature-adaptive triggering
threshold — evaluated on both a square and a circular reference path.

Built on ROS 2 Jazzy + Gazebo Harmonic.

**Note:** this repo covers the simulation study described below. Results
correspond to a 5-trial, Welch's-t-test-validated comparison of ZOH vs.
smart hold (and a curvature-adaptive threshold) on square and circular
reference trajectories.

## Key results

- σ = 0.1 cuts control-update rate ~90% vs. continuous control while
  keeping mean ATE under 1.1 cm (square path).
- **Smart hold significantly reduces ATE vs. ZOH** at both σ = 0.1 and
  σ = 0.2, on both the square (n=5, p < 0.001) and circle (n=5, p < 0.001)
  trajectories.
- On the circle (constant curvature, no corners), smart hold's event rate
  also drops to near zero (0.01–0.04 events/s) after an initial transient —
  a bandwidth benefit that did **not** reach significance on the square.
- The curvature-adaptive threshold helps at some square-path operating
  points but hit a startup-timing race condition on the circle path,
  reported as an open limitation rather than a quantified result.

## Layers

- **L1** — Fixed-σ event trigger, zero-order hold (`hold_mode=zoh`)
- **L2** — Smart hold: local dead-reckoning toward the live feedforward
  trajectory between events (`hold_mode=smart`)
- **L3** — Curvature-adaptive σ on top of L2 (`adaptive_sigma=true`) —
  works on the square path; has a known startup-timing race condition on
  the circle path (see Status below)
- **L4** — Hardware port (RPi5 + ESP32) + communication-dropout stress
  test (planned — see `docs/project_plan.pdf`)

## Packages

- **`etc_control/`** — trigger/control nodes (`trajectory_generator_node.py`,
  `event_trigger_node.py`, `esp32_emulator_node.py`) and the top-level
  experiment orchestrator (`run_experiment.launch.py`)
- **`swarm_description/`** — trimmed to only what `run_experiment.launch.py`
  actually pulls in: `gazebo.launch.py` (loads `etc_arena.sdf`) and
  `spawn.launch.py` (spawns the robot from `fwld.urdf.xacro` /
  `ros2_control.xacro`). Motion is driven natively by
  `gz-sim-diff-drive-system`; no `ros2_control` controller_manager is
  launched in this pipeline.

  **Note:** `swarm_description` also backs a separate multi-camera
  SLAM/swarm project (RTAB-Map, infra-camera point-cloud merging,
  multi-robot spawning, RViz config). That code is intentionally **not**
  included here — this repo only carries what the ETC experiments need
  to build and run standalone.

## Docs

- `docs/project_plan.pdf` — longer-term roadmap (hardware split L4,
  dropout testing, delay-augmented stability bound)

## Running an experiment

```bash
source ~/etc_ws/install/setup.bash
ros2 launch etc_control run_experiment.launch.py \
    shape:=square sigma:=0.1 hold_mode:=smart \
    adaptive_sigma:=false bag_name:=square_sigma0.1_smart_t1
```

## Analysis

Analysis scripts live in `analysis/` and read rosbags + event logs to
compute ATE (phase-aligned against the idealized reference trajectory) and
event rate. Run from a Python venv with `rosbags`, `numpy`, `matplotlib`
(and optionally `scipy` for Welch's t-test):

```bash
source plot_env/bin/activate
python3 analysis/analyze_experiment.py        # square: L1 vs L2
python3 analysis/analyze_repeated_trials.py    # square: repeated trials
python3 analysis/analyze_l3_comparison.py      # square: L1 vs L2 vs L3
python3 analysis/analyze_circle_pilot.py       # circle: pilot sweep
python3 analysis/analyze_circle_repeated.py    # circle: repeated trials + t-test
```

Rosbag trial data is **not** version-controlled (see `.gitignore`) — only
the code and resulting figures are.

## Results

Summary figures are in `results/figures/`:

- `fig_trajectory_comparison.png`, `fig_circle_trajectory_comparison.png`
- `fig_bandwidth_accuracy_tradeoff.png`
- `fig_zoh_vs_smart_hold.png`, `fig_zoh_vs_smart_hold_errorbars.png`,
  `fig_circle_zoh_vs_smart_errorbars.png`
- `fig_l3_three_way_comparison.png`, `fig_l3_events_comparison.png`

## Status

Simulation results for L1/L2/L3 on the square trajectory (repeated
trials, n=5) and circle trajectory (n=5, with Welch's t-test) are done and
written up for an ICRM submission (not included in this repo — see
below). Known open item: the curvature-adaptive threshold has an
unresolved startup-timing race condition on the circle path. Hardware
port (RPi5 + ESP32 over serial) and L4 dropout testing are next (see
`docs/project_plan.pdf`).

## Citation

If you use this code, a paper describing this work is under submission
to ICRM (S. C. Yadav, R. P. Kumar); citation details to follow once
accepted.
