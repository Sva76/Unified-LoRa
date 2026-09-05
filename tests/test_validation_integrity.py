"""Offline regression checks for scientific measurement errors; no API or GPU."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from validation.phi_utils import (PhiJumpAlarm, PhiJumpMonitor, completion_example,
                                  first_sustained_event, write_run_json)
from validation.revisql_prediction_analysis import EXPECTED, derive, load_raw
from validation.test7_metrics import analyze_run, summarize

ROOT = Path(__file__).resolve().parents[1]


class CompletionAlignmentTests(unittest.TestCase):
    def test_one_token_answer_is_predicted_from_prompt(self):
        inputs, targets, weights = completion_example([10, 20], [30])
        self.assertEqual(inputs, [10, 20])
        self.assertEqual(targets, [20, 30])
        self.assertEqual(weights, [0.0, 1.0])
        self.assertNotIn(30, inputs)

    def test_every_completion_token_is_supervised_once(self):
        inputs, targets, weights = completion_example([10, 20], [30, 40, 50])
        supervised = [(i, t) for i, t, w in zip(inputs, targets, weights) if w]
        self.assertEqual(supervised, [(20, 30), (30, 40), (40, 50)])
        self.assertEqual(len(inputs), len(targets))
        self.assertEqual(len(inputs), len(weights))

    def test_empty_sequences_require_explicit_handling(self):
        for prompt, answer in [([], [1]), ([1], []), ([], [])]:
            with self.subTest(prompt=prompt, answer=answer), self.assertRaises(ValueError):
                completion_example(prompt, answer)


class AlarmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "validation/phi_lead_time_log.json").read_text())

    def test_loss_only_monitor_reproduces_all_5000_historical_observations(self):
        count = 0
        for arm, runs in self.data["logs"].items():
            for seed, log in runs.items():
                monitor = PhiJumpMonitor()
                for step, loss, phi, _ in log:
                    actual = monitor.update(float("nan") if loss is None else loss)
                    self.assertAlmostEqual(actual, phi, delta=1e-10,
                                           msg=f"{arm}, seed {seed}, step {step}")
                    count += 1
        self.assertEqual(count, 5000)

    def test_historical_onsets_are_preserved_and_confirmation_is_later(self):
        old = {r["seed"]: r for r in self.data["collapse"]}
        report = summarize(self.data)
        self.assertFalse(report["confirmatory"])
        for row in report["collapse"]:
            saved = old[row["seed"]]
            self.assertEqual(row["t_c_onset"], saved["t_c"])
            self.assertEqual(row["t_a_phi_onset"], saved["t_a_phi"])
            self.assertEqual(row["legacy_lead_phi"], saved["lead_phi"])
            if row["t_a_phi_onset"] is not None:
                self.assertEqual(row["t_a_phi_confirmed"], row["t_a_phi_onset"] + 2)

    def test_online_alarm_matches_causal_offline_time(self):
        for runs in self.data["logs"].values():
            for seed, log in runs.items():
                alarm = PhiJumpAlarm()
                first_alarm = None
                for step, loss, _, _ in log:
                    if alarm.update(loss) and first_alarm is None:
                        first_alarm = step
                self.assertEqual(first_alarm, analyze_run(log)["t_a_phi_confirmed"], seed)

    def test_zero_false_alarms_does_not_apply_to_single_threshold_from_step_zero(self):
        self.assertEqual(summarize(self.data)["false_alarms"], 0)
        naive_fires = sum(any(p >= .1 for _, _, p, _ in log)
                          for log in self.data["logs"]["healthy"].values())
        self.assertEqual(naive_fires, 4)

    def test_confirmation_cannot_use_future_exceedances(self):
        self.assertIsNone(first_sustained_event([8, 9], [True, True], 3))
        event = first_sustained_event([8, 9, 10], [True, True, True], 3)
        self.assertEqual((event.onset, event.confirmed), (8, 10))

    def test_step_gaps_reset_confirmation_streak(self):
        event = first_sustained_event([8, 9, 12, 13, 14], [True] * 5, 3)
        self.assertEqual((event.onset, event.confirmed), (12, 14))

    def test_nonfinite_loss_is_not_a_healthy_online_observation(self):
        alarm = PhiJumpAlarm()
        for loss in [float("nan"), float("inf"), -float("inf")]:
            with self.assertRaises(ValueError):
                alarm.update(loss)
        self.assertEqual(alarm.step, -1)


class ReViSQLBaselineTests(unittest.TestCase):
    def frame(self, losses):
        return pd.DataFrame({"step": np.arange(len(losses)), "loss_per_token": losses,
                             "ppo_kl": np.arange(len(losses), dtype=float) / 10})

    def test_mad_uses_one_median_for_the_whole_window(self):
        df = self.frame(np.arange(1., 10.))
        corrected = derive(df)
        legacy = derive(df, "legacy-v1")
        self.assertEqual(corrected.rolling_mad5.iloc[-1], 1.)
        self.assertEqual(legacy.rolling_mad5.iloc[-1], 2.)
        self.assertEqual(corrected.rolling_mad5.first_valid_index(), 4)

    def test_mad_is_robust_to_a_single_large_outlier(self):
        result = derive(self.frame([1., 1., 1., 1., 1000.]))
        self.assertEqual(result.rolling_mad5.iloc[-1], 0.)
        self.assertGreater(result.rolling_std5.iloc[-1], 0.)

    def test_z_uses_only_the_preceding_five_values(self):
        result = derive(self.frame([1., 2., 3., 4., 5., 100.]))
        self.assertAlmostEqual(result.abs_causal_z.iloc[-1], 97. / np.sqrt(2.))
        self.assertTrue(result.abs_causal_z.iloc[:5].isna().all())

    def test_constant_history_z_is_undefined_not_infinite(self):
        result = derive(self.frame([0.] * 5 + [100.]))
        self.assertTrue(result.abs_causal_z.isna().all())

    def test_future_mutation_cannot_change_current_predictors(self):
        original = self.frame([1., 3., 2., 5., 4., 7., 6., 8., 9., 10.])
        changed = original.copy()
        changed.loc[7:, "loss_per_token"] = 1e9
        predictors = ["phi_jump", "phi_abs", "abs_delta_loss", "rolling_std5",
                      "rolling_mad5", "abs_causal_z"]
        pd.testing.assert_frame_equal(derive(original).loc[:6, predictors],
                                      derive(changed).loc[:6, predictors])

    def test_target_horizons_preserve_unavailable_tail(self):
        result = derive(self.frame(np.arange(8, dtype=float)))
        self.assertEqual(result.future_ppo_kl_t5.iloc[0], .5)
        self.assertTrue(result.future_ppo_kl_t5.iloc[-5:].isna().all())

    def test_versions_do_not_change_phi_or_targets(self):
        df = self.frame([0., 2., -1., 5., 0., 3., -4., 2., 1.])
        cols = ["phi_jump", "phi_abs", "future_ppo_kl_t1", "future_ppo_kl_t2",
                "future_ppo_kl_t5"]
        pd.testing.assert_frame_equal(derive(df)[cols], derive(df, "legacy-v1")[cols])

    def test_missing_loss_cannot_silently_become_zero_innovation(self):
        with self.assertRaises(ValueError):
            derive(self.frame([0., float("nan"), 2.]))

    def test_unknown_derivation_version_is_rejected(self):
        with self.assertRaises(ValueError):
            derive(self.frame([1.]), "typo")


class RawIntegrityTests(unittest.TestCase):
    def record(self, step):
        row = {key: None for key in EXPECTED}
        row.update(step=step, action_tokens=2, loss_sum=-1., loss_per_token=-.5, ppo_kl=.1)
        return row

    def load_rows(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return load_raw(path)

    def test_signed_and_zero_losses_are_retained(self):
        rows = [self.record(0), self.record(1)]
        rows[1].update(loss_sum=0., loss_per_token=0.)
        self.assertEqual(self.load_rows(rows).loss_per_token.tolist(), [-.5, 0.])

    def test_bad_raw_trajectories_are_rejected(self):
        cases = [[], [self.record(0), self.record(0)], [self.record(0), self.record(2)]]
        for field, value in [("step", .5), ("loss_per_token", float("nan")),
                             ("action_tokens", 0), ("action_tokens", 1.5),
                             ("loss_per_token", 999.)]:
            row = self.record(0)
            row[field] = value
            cases.append([row])
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.load_rows(rows)

    def test_new_results_cannot_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            write_run_json(path, {"loss": float("nan")})
            original = path.read_bytes()
            self.assertEqual(json.loads(original), {"loss": None})
            with self.assertRaises(FileExistsError):
                write_run_json(path, {"loss": 0.})
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
