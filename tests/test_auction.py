import unittest

from reverse_auction_assignment import CostWeights, RobotRequest, Worker, assign, pair_cost, solve_auction


class ReverseAuctionTests(unittest.TestCase):
    def test_nearest_pairs_are_selected(self):
        workers = [Worker("w0", (0, 0)), Worker("w1", (8, 0))]
        requests = [RobotRequest("r0", (1, 0)), RobotRequest("r1", (7, 0))]
        self.assertEqual(assign(workers, requests), [("w0", "r0"), ("w1", "r1")])

    def test_assignment_is_one_to_one(self):
        result = solve_auction([[1, 2], [1, 3]])
        self.assertEqual(len({index for index in result if index is not None}), 2)

    def test_fewer_requests_use_dummy_objects(self):
        workers = [Worker("w0", (0, 0)), Worker("w1", (2, 0))]
        requests = [RobotRequest("r0", (1, 0))]
        result = assign(workers, requests)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "r0")

    def test_zone_penalty_is_explicit(self):
        worker = Worker("w0", (0, 0), frozenset({10}))
        inside = RobotRequest("inside", (3, 0), node_id=10)
        outside = RobotRequest("outside", (1, 0), node_id=20)
        weights = CostWeights(zone_mismatch=5)
        self.assertLess(pair_cost(worker, inside, weights), pair_cost(worker, outside, weights))

    def test_custom_distance_is_supported(self):
        workers = [Worker("w0", (0, 0))]
        requests = [RobotRequest("r0", (9, 9))]
        self.assertEqual(assign(workers, requests, distance=lambda _a, _b: 4), [("w0", "r0")])

    def test_invalid_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_auction([[1], [2, 3]])


if __name__ == "__main__":
    unittest.main()
