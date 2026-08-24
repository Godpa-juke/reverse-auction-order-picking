"""Order generation helpers used by legacy CLI scripts."""

from __future__ import annotations

from typing import List, Sequence


def order_gen(points_of_interest: int, rack_num: int, routing_nodes: Sequence[Sequence[int]]) -> list[list[int]]:
    """Create a randomized order list ensuring unique rack selections."""

    import random  # local import to keep module lightweight

    rack_list: list[int] = []
    for _ in range(points_of_interest):
        rack = random.randint(1, rack_num)
        while rack in rack_list:
            rack = random.randint(1, rack_num)
        rack_list.append(rack)

    order_list: list[list[int]] = []
    for rack in rack_list:
        node = 0
        for node_idx, racks in enumerate(routing_nodes):
            if rack in racks:
                node = node_idx
        order_list.append([rack, 1, node])

    return order_list


def order_modi(order_list: List[List[int]], pcs: int) -> List[List[int]]:
    """Increase SKU quantity for the first ``pcs`` entries of an order list."""

    for idx in range(min(pcs, len(order_list))):
        order_list[idx][1] += 1
    return order_list
