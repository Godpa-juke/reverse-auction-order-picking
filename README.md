# Reverse-Auction Assignment for Human–Robot Order Picking

A small, data-free reference implementation of one-to-one worker-to-robot assignment using a Bertsekas-style auction solver.

This repository explains and implements the **reactive assignment baseline** used in collaborative order-picking research: eligible workers bid for serviceable robot requests using a configurable pairwise cost. The code is independent of any warehouse layout, order file, simulator output, manuscript, or measured result.

## What is included

- immutable worker and robot-request inputs;
- configurable travel, wait, service, and zone-mismatch costs;
- a rectangular one-to-one auction solver;
- deterministic tie-breaking;
- toy unit tests for assignment invariants.

## What is not included

- papers or PDFs;
- LaTeX sources, tables, figures, or reported numbers;
- warehouse layouts or order workloads;
- raw or aggregate experiment data;
- private simulator code or Git history.

## Install

```bash
python -m pip install -e .
```

## Example

```python
from reverse_auction_assignment import (
    CostWeights,
    RobotRequest,
    Worker,
    assign,
)

workers = [
    Worker("w0", position=(0, 0)),
    Worker("w1", position=(8, 0)),
]
requests = [
    RobotRequest("r0", position=(1, 0)),
    RobotRequest("r1", position=(7, 0)),
]

pairs = assign(workers, requests, weights=CostWeights())
print(pairs)
# [('w0', 'r0'), ('w1', 'r1')]
```

## Cost model

For a worker-request pair, the default cost is:

```text
travel_weight × Manhattan distance
+ robot_wait_weight × accumulated robot wait
+ service_weight × expected service duration
+ zone penalty when the request lies outside the worker's allowed nodes
```

Applications can supply their own distance function without changing the auction solver.

## Test

```bash
python -m unittest discover -s tests -v
```
