# Reverse-Auction Order Picking

Executable research code for **global human-to-robot assignment in collaborative order-picking systems**. This repository contains the actual warehouse simulator integration—not a standalone pseudocode translation or a paper file listing.

The implementation extends [RWARE](https://github.com/semitable/robotic-warehouse) with human pickers, robot requests, obstacle-aware travel costs, deterministic Bertsekas-style auction matching, and guarded re-auction.

## Canonical Site-A experiment environment

![Canonical Site-A AHEAD 45-degree Isaac Sim rollout](media/canonical_sitea_ahead_45deg.gif)

This shared environment demo mirrors the canonical comparison setup: Site-A batched orders, Batch-Random sequencing, 8 human pickers, and 20 robots. The displayed motion is produced by the real RWARE event loop using the AHEAD `rv_static` policy and rendered by an Isaac Sim/Isaac Lab `Camera` at a fixed 45-degree downward angle; **it is an environment-level demo, not a Reverse Auction result**. The method-specific Reverse Auction rollout remains below. The operational orders, facility trace, and layout data are not included in this public repository.

## Isaac Sim rollout

![Reverse Auction actual Isaac Sim rollout](media/reverse_auction_isaac_sim.gif)

The looping GIF is derived from RGB captured by an actual Isaac Sim/Isaac Lab `Camera` on a procedural USD warehouse stage. It executes this repository's `AuctionAssignmentStrategy` cost matrix and Bertsekas solver. Robots first reach their pick targets; only then does the reactive auction dispatch the two pickers. The title/status overlay is applied to the captured Isaac RGB frames.

```bash
REPO="$PWD"
cd /path/to/IsaacLab
PYTHONPATH="$REPO:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks" \
  ./isaaclab.sh -p "$REPO/scripts/render_isaac_warehouse.py" \
  --method auction --headless --enable_cameras --device cuda:0 \
  --output "$REPO/media/reverse_auction_isaac_sim.mp4" \
  --poster "$REPO/media/reverse_auction_isaac_sim.png"
```

The machine-readable run receipt is [`media/reverse_auction_isaac_sim.json`](media/reverse_auction_isaac_sim.json). This is a qualitative synthetic-layout demonstration, not a private-facility throughput experiment.

## What you can run

After cloning, you can:

1. inspect the exact bid-cost construction used by the simulator;
2. compare auction matching with a greedy solver on the same value matrix;
3. instantiate the synthetic RWARE map used by the contract tests;
4. run the assignment, map-DSL, and engine-enforcement tests without private data.

## Quick start

Python 3.10–3.13 is supported.

```bash
git clone https://github.com/Godpa-juke/reverse-auction-order-picking.git
cd reverse-auction-order-picking
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/run_synthetic_assignment.py
pytest
```

The example prints a real cost matrix produced by `AuctionAssignmentStrategy` and the robot index selected for each human by both auction and greedy solvers. It is deterministic and needs no warehouse dataset.

## Method implemented here

For each candidate human `h` and robot request `r`, the simulator builds a cost containing:

- obstacle-aware human travel distance;
- estimated robot service time;
- robot waiting-time urgency;
- human waiting-time fairness;
- an optional out-of-zone penalty.

The solver maximizes the negative cost under one-to-one assignment. Re-auction is guarded by:

- **arrival lock** `tau_lock`: do not switch an assignment near arrival;
- **minimum gain** `delta_gain`: switch only when the cost improvement is material;
- **switch budget** `max_reassign`: cap reassignment count per request.

Ablation strategies keep the same simulator path while independently removing urgency, fairness, service cost, zone handling, re-auction, or the auction solver itself.

## Code map

| Path | Purpose |
|---|---|
| `rware/engine/human_assignment.py` | snapshots, cost matrix, auction/greedy solvers, re-auction guards, strategy registry |
| `rware/engine/warehouse_engine.py` | live simulator integration and assignment refresh |
| `rware/core/map_dsl.py` | synthetic/public map parser and traffic constraints |
| `rware/warehouse.py` | multi-agent warehouse environment |
| `rware/algorithm/path_planning/` | A*, JPS, and modified JPS planners |
| `scripts/run_synthetic_assignment.py` | deterministic no-data example |
| `scripts/render_isaac_warehouse.py` | actual Isaac Sim USD-stage rollout and Camera capture |
| `tests/test_auction_ablation.py` | solver and cost-term contracts |
| `tests/test_engine_enforcement.py` | inline-map movement constraints |

## Public reproducibility boundary

Included:

- executable simulator and assignment source;
- an inline synthetic warehouse layout;
- deterministic no-data example;
- behavioral tests;
- license, citation, and source provenance.

Not included:

- real warehouse orders or operational records;
- facility-specific maps and precomputed path arrays;
- raw experiment outputs, checkpoints, manuscripts, or internal paths.

The repository therefore supports **algorithm inspection, extension, synthetic execution, and unit-level reproduction**. It does not claim to reproduce private-site throughput numbers from public inputs.

## Tests

```bash
pytest
```

The public test set is intentionally asset-independent. It executes behavior rather than checking source text:

- auction versus greedy assignment;
- exact ablation cost relationships;
- strategy registration and re-auction overrides;
- map-overlay parsing;
- robot/human movement constraints on an inline synthetic map.

## Extending the artifact

New assignment policies implement `HumanAssignmentStrategy` and register through `@register_strategy`. To compare policies fairly, use the shared obstacle-aware distance helpers instead of introducing a policy-specific distance proxy.

## Provenance and license

See [`PROVENANCE.md`](PROVENANCE.md) for the canonical export revision. This project is a modified derivative of RWARE. The upstream MIT copyright and license are preserved in [`LICENSE`](LICENSE).
