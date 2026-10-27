"""Authors: Aritra Saha and Ugochukwu Njoku.
Paper: Survival Mixture Density Networks, MLHC 2022.
https://proceedings.mlr.press/v182/han22a.html

Exercise the actual portable benchmark adapters, CLI paths and saved artifacts.
"""

import contextlib
import importlib.util
import io
import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.survival_benchmark import run_baselines, run_survival_mdn
from examples.survival_benchmark.baselines import CoxAdapter, DeepSurvAdapter
from examples.survival_benchmark.calibrate_saved import calibrate
from tests.core.survival_synthetic import make_survival_samples

HAS_COX = importlib.util.find_spec("sksurv") is not None
HAS_DEEPSURV = all(
    importlib.util.find_spec(name) is not None for name in ("pycox", "torchtuples")
)


def sample_arrays(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a small existing PyHealth synthetic survival fixture."""
    rows = make_survival_samples(n=n, seed=seed)
    return np.array([r["features"] for r in rows], np.float32), np.array(
        [r["survival"] for r in rows], np.float32
    )


class TestSurvivalBenchmarkInputs(unittest.TestCase):
    """Check boundaries and source identity without optional backend packages."""

    def test_safe_batches(self):
        self.assertEqual(run_survival_mdn.safe_batch_size(3, 2), 3)
        self.assertEqual(run_survival_mdn.safe_batch_size(513, 512), 511)
        for n in range(2, 40):
            size = run_survival_mdn.safe_batch_size(n, 8)
            self.assertGreaterEqual(size, 2)
            self.assertNotEqual(n % size, 1)
        for n, requested in [(1, 8), (3, 1), (0, 8)]:
            with self.subTest(n=n), self.assertRaises(ValueError):
                run_survival_mdn.safe_batch_size(n, requested)

    def test_synthetic_membership_and_shapes(self):
        benchmark = run_survival_mdn.SyntheticBenchmark()
        ids, samples, metadata = benchmark.split(1)
        self.assertEqual([len(ids[n]) for n in run_survival_mdn.SPLITS], [120, 30, 30])
        self.assertEqual(
            len({patient for group in ids.values() for patient in group}), 180
        )
        self.assertEqual(ids, benchmark.split(1)[0])
        self.assertNotEqual(ids, benchmark.split(2)[0])
        self.assertTrue(metadata["synthetic"])
        for part in run_survival_mdn.split_parts(samples, ids, 1):
            x, y, patients = run_baselines.arrays(part)
            self.assertEqual(x.shape, (len(part), 7))
            self.assertEqual(y.shape, (len(part), 2))
            self.assertEqual(len(patients), len(part))

    def test_saved_calibration_requires_exact_values_and_events(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prediction.npz"
            np.savez(path, event=[1], survival=[[1.0, 0.5]])
            with self.assertRaisesRegex(ValueError, "Exact observed-time"):
                calibrate(path)
            np.savez(path, survival_at_observed_time=[0.5])
            with self.assertRaisesRegex(ValueError, "event flags"):
                calibrate(path)
            np.savez(path, survival_at_observed_time=[0.25, 0.75], event=[1, 1])
            self.assertEqual(calibrate(path, bins=2)["bin_counts"], [1.0, 1.0])

    def test_resume_rejects_changed_settings(self):
        expected = {"dataset": "gbsg", "epochs": 100, "source_hash": "a"}
        run_survival_mdn.validate_resume({"request": expected}, expected)
        for changed in [
            {},
            {**expected, "epochs": 5},
            {**expected, "source_hash": "b"},
        ]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                run_survival_mdn.validate_resume({"request": changed}, expected)

    def test_output_roots_cannot_mix_datasets_or_runners(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            run_survival_mdn.prepare_output(path, "gbsg", "mdn")
            run_survival_mdn.prepare_output(path, "gbsg", "mdn")
            for dataset, runner in [("support2", "mdn"), ("gbsg", "baselines")]:
                with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                    run_survival_mdn.prepare_output(path, dataset, runner)

    def test_real_cli_requires_paths(self):
        for main in [run_baselines.main, run_survival_mdn.main]:
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                main(["--dataset", "gbsg", "--out", "/unused"])
            self.assertEqual(error.exception.code, 2)

    def test_adapter_configuration_rejects_silent_integer_truncation(self):
        for config in [
            {"epochs": 1.5},
            {"patience": True},
            {"seed": -1},
            {"hidden": [3.5]},
        ]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                DeepSurvAdapter(**config)

    def test_source_fingerprints_include_actual_native_modules(self):
        hashes = run_survival_mdn.implementation_hashes()
        self.assertIn("baselines.py", hashes)
        self.assertIn("pyhealth/metrics/survival_curve.py", hashes)
        self.assertIn("pyhealth/datasets/support2.py", hashes)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))


class TestSurvivalBenchmarkMDN(unittest.TestCase):
    """Run the public synthetic MDN entry point and inspect its real outputs."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.path = Path(cls.directory.name) / "mdn"
        cls.args = [
            "--dataset",
            "synthetic",
            "--splits",
            "1",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--out",
            str(cls.path),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            run_survival_mdn.main(cls.args)
        cls.record_path = cls.path / "mdn_K10_softplus/split_1/seed_1/record.json"

    def test_mdn_outputs_and_exact_calibration(self):
        record = json.loads(self.record_path.read_text())
        self.assertEqual(record["counts"], {"train": 120, "valid": 30, "test": 30})
        self.assertTrue(np.isfinite(record["val_nll"]))
        self.assertTrue(all(np.isfinite(v) for v in record["test_metrics"].values()))
        self.assertEqual(
            record["d_calibration"],
            calibrate(self.record_path.parent / "predictions.npz"),
        )
        self.assertIn("trainer/best.ckpt", record["artifacts"])
        with np.load(
            self.record_path.parent / "predictions.npz", allow_pickle=False
        ) as data:
            self.assertEqual(data["survival_at_observed_time"].shape, (30,))
            self.assertEqual(len(set(data["patient_id"].tolist())), 30)

    def test_matching_cli_resume_does_not_rewrite_fit(self):
        original = self.record_path.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            run_survival_mdn.main(self.args)
        self.assertEqual(original, self.record_path.read_bytes())

    def test_changed_cli_and_missing_checkpoint_are_rejected(self):
        changed = self.args.copy()
        changed[changed.index("--epochs") + 1] = "2"
        with self.assertRaises(ValueError):
            run_survival_mdn.main(changed)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "copy"
            shutil.copytree(self.path, target)
            (target / "mdn_K10_softplus/split_1/seed_1/trainer/best.ckpt").unlink()
            copied = self.args.copy()
            copied[-1] = str(target)
            with self.assertRaisesRegex(ValueError, "Missing or changed"):
                run_survival_mdn.main(copied)

    def test_preexisting_trainer_handler_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            handler = logging.FileHandler(Path(folder) / "external.log")
            logger = run_survival_mdn.trainer_logger
            logger.addHandler(handler)
            try:
                args = self.args.copy()
                args[-1] = str(Path(folder) / "fit")
                with contextlib.redirect_stdout(io.StringIO()):
                    run_survival_mdn.main(args)
                self.assertIn(handler, logger.handlers)
                self.assertIsNotNone(handler.stream)
            finally:
                logger.removeHandler(handler)
                handler.close()


class TestSurvivalBenchmarkAdapters(unittest.TestCase):
    """Exercise each installed optional baseline against the shared evaluator."""

    def setUp(self):
        self.train, self.valid, self.test = [
            sample_arrays(n, seed) for n, seed in [(100, 4), (30, 5), (30, 6)]
        ]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)

    def exercise(self, adapter):
        adapter.fit(
            *self.train, validation=self.valid, checkpoint_dir=self.path / "best"
        )
        x, y = self.test
        grid = np.linspace(0, 20, 11)
        predictions = adapter.predict_survival(x, grid)
        self.assertEqual(predictions.shape, (len(x), len(grid)))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue(((predictions >= 0) & (predictions <= 1)).all())
        self.assertTrue((np.diff(predictions, axis=1) <= 1e-6).all())
        np.testing.assert_equal(predictions[:, 0], np.ones(len(x)))
        np.testing.assert_allclose(
            adapter.predict_survival(x[::-1], grid),
            predictions[::-1],
            rtol=1e-6,
            atol=1e-6,
        )
        adapter.save(self.path / "saved")
        rebuilt = type(adapter).load(self.path / "saved")
        np.testing.assert_array_equal(rebuilt.predict_survival(x, grid), predictions)
        metrics, reference, calibration, observed = run_baselines.score(adapter, x, y)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertTrue(all(np.isfinite(v) for v in reference.values()))
        self.assertAlmostEqual(sum(calibration["bin_counts"]), len(x))
        path = self.path / "predictions.npz"
        np.savez(path, event=y[:, 1], survival_at_observed_time=observed)
        self.assertEqual(calibrate(path), calibration)

    @unittest.skipUnless(
        HAS_COX, "Install optional survival benchmark requirements for Cox"
    )
    def test_cox_contract_and_reload(self):
        self.exercise(CoxAdapter())

    @unittest.skipUnless(
        HAS_DEEPSURV, "Install optional survival benchmark requirements for DeepSurv"
    )
    def test_deepsurv_contract_and_reload(self):
        self.exercise(DeepSurvAdapter(hidden=[8], epochs=4, patience=2, seed=3))

    @unittest.skipUnless(
        HAS_COX, "Install optional survival benchmark requirements for Cox"
    )
    def test_training_hazards_do_not_use_validation_outcomes(self):
        a = CoxAdapter().fit(*self.train, validation=self.valid)
        x, y = self.valid
        altered = y.copy()
        altered[:, 0] *= 100
        b = CoxAdapter().fit(*self.train, validation=(x, altered))
        np.testing.assert_array_equal(
            a.predict_survival(x, [1.0, 5.0]), b.predict_survival(x, [1.0, 5.0])
        )

    def test_unknown_model_is_rejected_before_loading_data(self):
        with self.assertRaises(ValueError):
            run_baselines.run_fit("unknown", {}, (), self.path)

    @unittest.skipUnless(
        HAS_COX, "Install optional survival benchmark requirements for Cox"
    )
    def test_baseline_cli_and_resume(self):
        args = [
            "--dataset",
            "synthetic",
            "--models",
            "cox",
            "--splits",
            "1",
            "--out",
            str(self.path),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            run_baselines.main(args)
        path = self.path / "cox/split_1/seed_1/record.json"
        original = path.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            run_baselines.main(args)
        self.assertEqual(path.read_bytes(), original)
        config = self.path / "config.json"
        config.write_text('{"cox": {"alpha": 1.0}}')
        with self.assertRaises(ValueError):
            run_baselines.main([*args, "--config", str(config)])


if __name__ == "__main__":
    unittest.main()
