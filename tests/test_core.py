import unittest

from miniport.rewards import Checkpoint, RewardTracker
from miniport.tasks import GATES, heldout_tasks, training_tasks


def checkpoint(**overrides):
    fields = dict(action="test", command="python test.py", before_hash="a", after_hash="a",
                  result_signature="failure:layer", passed_gates=GATES[:2], diagnostic="layer:shape",
                  remaining_actions=10)
    fields.update(overrides)
    return Checkpoint(**fields)


class CoreTests(unittest.TestCase):
    def test_split(self):
        train, held = training_tasks(), heldout_tasks()
        self.assertEqual((len(train), len(held)), (18, 6))
        self.assertFalse({t.id for t in train} & {t.id for t in held})
        self.assertFalse({t.width for t in train} & {t.width for t in held})

    def test_repetition_and_new_diagnostic(self):
        tracker = RewardTracker()
        tracker.score(checkpoint())
        self.assertIn("repetition", tracker.score(checkpoint())["events"])
        self.assertNotIn("repetition", tracker.score(checkpoint(diagnostic="layer:bias"))["events"])
        self.assertNotIn("repetition", tracker.score(checkpoint(before_hash="a", after_hash="b"))["events"])

    def test_regression_cannot_farm_gate_reward(self):
        tracker = RewardTracker()
        tracker.score(checkpoint())
        self.assertEqual(tracker.score(checkpoint(passed_gates=GATES[:1]))["events"]["regression"], -0.75)
        self.assertNotIn("new_gate", tracker.score(checkpoint())["events"])

    def test_stop_only_when_actionable_and_budget_remains(self):
        for changes, expected in (({}, True), ({"remaining_actions": 0}, False),
                                  ({"environment_outage": True}, False),
                                  ({"actionable_failure": False}, False),
                                  ({"passed_gates": GATES}, False)):
            result = RewardTracker().score(checkpoint(action="stop", **changes))
            self.assertEqual("premature_termination" in result["events"], expected)

    def test_metric_comparability_and_bound(self):
        tracker = RewardTracker()
        tracker.score(checkpoint(error=100, metric_name="layer"))
        result = tracker.score(checkpoint(error=0, metric_name="layer"))
        self.assertEqual(result["events"]["numerical_change"], 0.25)
        self.assertNotIn("numerical_change", tracker.score(checkpoint(error=3, metric_name="e2e"))["events"])

    def test_invalid_gate_order(self):
        with self.assertRaises(ValueError):
            RewardTracker().score(checkpoint(passed_gates=("layer",)))

    def test_metric_noise_not_rewarded(self):
        tracker = RewardTracker()
        tracker.score(checkpoint(error=1.0, metric_name="layer"))
        result = tracker.score(checkpoint(error=0.99999, metric_name="layer"))
        self.assertNotIn("numerical_change", result["events"])


if __name__ == "__main__":
    unittest.main()
