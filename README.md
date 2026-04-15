# handover-env

Raylib-based cellular handover simulator with interactive visualization.

The project models UE mobility, RSRP-driven handovers, SON parameter tuning, and an interactive renderer for debugging handover behavior.

![Handover Environment Screenshot](handover-env.png)

## Features

- 5G baseline simulation config (`24.25 GHz`, configurable TX power and thresholds)
- Naive handover algorithm with hysteresis margin + time-to-trigger (TTT)
- Handover classification metrics: early, late, ping-pong, success
- Link termination handling when no serving BS is valid
- SON tuning for `HYSTERISIS_MARGIN` and `TIME_TO_TRIGGER` on failure events
- Interactive Raylib UI with:
  - coverage regions
  - UE/BS icon rendering from PNG assets
  - hover tooltips
  - pin panels (UE/BS)
  - flashing connection diagnostics (red=failure, green=successful handover)

## Project Index

| Path | Purpose |
|---|---|
| `main.py` | CLI entrypoint and scenario runner |
| `simulation/config.py` | Core simulation config dataclass and defaults |
| `simulation/simulator.py` | Main simulation step loop and SON trigger logic |
| `simulation/scenario_configs.py` | Prebuilt linear/realistic scenarios and network factories |
| `simulation/statistics.py` | Statistics counters |
| `algorithms/handover.py` | Handover decision algorithm (`naive_handover`) |
| `core/ue_bs_helpers.py` | Mobility updates, RSRP calculation, handover execution routing |
| `core/handover_helpers.py` | Handover type detection and handover execution helpers |
| `core/son.py` | SON tuning strategy |
| `entities/` | Domain entities (`UE`, `BaseStation`, `Network`, handover state, policy) |
| `rendering/renderer.py` | Main renderer and interaction handling |
| `rendering/entities.py` | UE/BS icon drawing + connection lines |
| `rendering/config.py` | Rendering config (colors, icon scales, sizes) |
| `rendering/statistics.py` | Statistics/config UI panels |
| `loggers/` | Structured event/error logging helpers |
| `bs.png`, `ue.png` | Icon assets used by renderer |
| `handover-env.png` | Project screenshot used in this README |

## Requirements

- Python `>=3.14`
- GUI-capable environment (Raylib windowing)

Dependencies (from `pyproject.toml`):

- `numpy>=2.4.3`
- `raylib>=5.5.0.4`

## Setup

```bash
git clone <your-repo-url>
cd handover-env
uv sync
```

If your shell does not auto-activate environments:

```bash
source .venv/bin/activate
```

## Running the Simulator

Default run (`realistic` scenario, up to 1000 steps):

```bash
python3 main.py
```

Run explicit scenario:

```bash
python3 main.py linear
python3 main.py realistic
python3 main.py test
```

Run for a fixed number of steps:

```bash
python3 main.py realistic --max-steps 2000
```

Run until interrupted:

```bash
python3 main.py realistic --run-forever
```

Stop with `Ctrl+C`, `ESC`, or by closing the window.

## UI Controls

- Hover UE/BS: show tooltip
- Left click UE: pin UE panel
- Left click BS: pin BS panel
- Right click pinned panel: unpin that panel

Pin panels display serving/connection context and are useful for live debugging.

## Scenarios

### `linear`

- Pure linear movement
- Spread-out BS and UE topology
- Different UE speeds

Defaults: `8 BS`, `12 UE`.

### `realistic`

- Mixed random + linear UE movement
- Jittered BS placement
- Heterogeneous BS TX powers

Defaults: `9 BS`, `14 UE`.

### `test`

- Small deterministic network (`3 BS`, `2 UE`) for quick sanity checks.

## Simulation Model Notes

- Path-loss model uses reference-distance + exponent form.
- 5G baseline defaults:
  - `DEFAULT_FREQUENCY = 24.25e9`
  - `DEFAULT_TX_POWER = 53.0`
  - `RLF_FAILURE_THRESHOLD = -97`
- Link is terminated when no BS has acceptable signal quality.

## Metrics

Tracked in `SimulationStatistics`:

- `early_handover_count`
- `late_handover_count`
- `ping_pong_handover_count`
- `successful_handover_count`

Late failures also include disconnection-driven failures.

## SON Behavior

When new failure events occur, SON updates:

- `HYSTERISIS_MARGIN`
- `TIME_TO_TRIGGER`

Config panel shows current and original values:

- Green text: increased vs original
- Red text: decreased vs original

## Rendering Notes

- Theme: white/black/gray base for clarity
- Connection overlays:
  - Red flashing line: failure condition
  - Green flashing line: recent successful handover
- UE/BS labels are rendered below icon sprites for readability

Icon scaling is configurable in `RendererConfig`:

- `bs_icon_scale`
- `ue_icon_scale`

## Customization Quick Guide

- Simulation behavior: `simulation/config.py`, `simulation/scenario_configs.py`
- Handover algorithm: `algorithms/handover.py`
- SON policy: `core/son.py`
- Visuals and icon scales: `rendering/config.py`
- Renderer interactions: `rendering/renderer.py`

## Notes

- This project is intentionally modular: algorithm, simulator, rendering, and env wrappers are decoupled.
- The current handover policy is a baseline naive algorithm intended to be replaced by improved or learned policies.
