"""Tests for SurvivalMDN and its synthetic survival fixture.

All data is synthetic and generated in memory; no downloads.
"""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SurvivalMDN
from tests.core.survival_synthetic import (
    BETA0,
    make_survival_samples,
    true_survival,
)

INPUT_SCHEMA = {"features": "tensor"}
OUTPUT_SCHEMA = {"survival": "tensor"}


def _dataset(samples, name="synthetic_survival"):
    return create_sample_dataset(
        samples=samples,
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        dataset_name=name,
    )


class TestSurvivalSyntheticFixture(unittest.TestCase):
    """The shared fixture produces contract-shaped, reproducible data."""

    def test_reproducible(self):
        a = make_survival_samples(n=20, seed=3)
        b = make_survival_samples(n=20, seed=3)
        self.assertEqual(a, b)

    def test_labels_are_valid(self):
        samples = make_survival_samples(n=500, censoring_rate=0.4, seed=0)
        times = np.array([s["survival"][0] for s in samples])
        events = np.array([s["survival"][1] for s in samples])
        self.assertTrue(np.all(np.isfinite(times)) and np.all(times > 0))
        self.assertTrue(set(np.unique(events)) <= {0.0, 1.0})
        self.assertAlmostEqual(1 - events.mean(), 0.4, delta=0.03)

    def test_no_censoring(self):
        samples = make_survival_samples(n=50, censoring_rate=0.0, seed=0)
        self.assertTrue(all(s["survival"][1] == 1.0 for s in samples))

    def test_true_survival(self):
        s = true_survival([[0.0, 0.0, 0.0]], [1e-6, np.exp(BETA0), 1e6])
        np.testing.assert_allclose(s[0], [1.0, 0.5, 0.0], atol=1e-6)


class TestSurvivalContract(unittest.TestCase):
    """Synthetic samples pass through PyHealth as the T1 contract expects."""

    def setUp(self):
        self.samples = make_survival_samples(n=8, n_features=3, seed=0)
        self.dataset = _dataset(self.samples)

    def test_batch_shapes_and_order(self):
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["features"].shape), (4, 3))
        self.assertEqual(tuple(batch["survival"].shape), (4, 2))
        self.assertTrue(torch.is_floating_point(batch["survival"]))
        expected = torch.tensor([s["survival"] for s in self.samples[:4]])
        torch.testing.assert_close(batch["survival"], expected)

    def test_tensor_label_gives_no_mode(self):
        model = SurvivalMDN(self.dataset, num_components=3)
        self.assertIsNone(model.mode)
        self.assertEqual(model.feature_key, "features")
        self.assertEqual(model.label_key, "survival")

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            SurvivalMDN(self.dataset, transform="sigmoid")
        with self.assertRaises(ValueError):
            SurvivalMDN(self.dataset, num_components=0)
        with self.assertRaises(ValueError):
            SurvivalMDN(self.dataset, time_scale=0.0)


def _batch(dataset, batch_size=64):
    return next(iter(get_dataloader(dataset, batch_size=batch_size, shuffle=False)))


class TestSurvivalMDNModel(unittest.TestCase):
    """Shapes, numerics and correctness of the model's math."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.samples = make_survival_samples(n=64, censoring_rate=0.4, seed=0)
        cls.dataset = _dataset(cls.samples)
        cls.batch = _batch(cls.dataset)
        cls.configs = [
            {"transform": "softplus", "num_components": 5},
            {"transform": "exp", "num_components": 5},
            {"transform": "softplus", "num_components": 1},
            {"transform": "exp", "num_components": 1},
        ]

    def _model(self, **kwargs):
        kwargs.setdefault("time_scale", 7.0)
        return SurvivalMDN(self.dataset, **kwargs)

    def test_forward_outputs(self):
        model = self._model(num_components=5)
        out = model(**self.batch)
        self.assertEqual(
            sorted(out), ["log_weights", "loss", "means", "scales", "y_true"]
        )
        self.assertEqual(out["loss"].dim(), 0)
        for key in ("log_weights", "means", "scales"):
            self.assertEqual(tuple(out[key].shape), (64, 5))
        min_scale = torch.nn.functional.softplus(torch.tensor(0.01))
        self.assertTrue((out["scales"] >= min_scale - 1e-6).all())
        self.assertTrue((out["means"].abs() <= 30).all())
        torch.testing.assert_close(
            out["log_weights"].exp().sum(-1), torch.ones(64), rtol=0, atol=1e-5
        )
        torch.testing.assert_close(out["y_true"], self.batch["survival"])

    def test_finite_loss_and_gradients(self):
        survival = self.batch["survival"]
        cases = {
            "mixed": survival,
            "all_events": torch.stack(
                [survival[:, 0], torch.ones_like(survival[:, 0])], 1
            ),
            "all_censored": torch.stack(
                [survival[:, 0], torch.zeros_like(survival[:, 0])], 1
            ),
        }
        for config in self.configs:
            for name, labels in cases.items():
                with self.subTest(**config, labels=name):
                    model = self._model(**config)
                    batch = dict(self.batch, survival=labels)
                    loss = model(**batch)["loss"]
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    grads = [
                        p.grad
                        for name, p in model.named_parameters()
                        if name != "_dummy_param"
                    ]
                    self.assertTrue(all(g is not None for g in grads))
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_survival_curves_bounded_and_monotone(self):
        t_grid = torch.cat([torch.tensor([1e-4, 1e-2]), torch.linspace(0.1, 500, 200)])
        for config in self.configs:
            with self.subTest(**config):
                model = self._model(**config)
                curves = model.predict_survival(self.batch["features"], t_grid)
                self.assertEqual(tuple(curves.shape), (64, len(t_grid)))
                self.assertTrue(((curves >= 0) & (curves <= 1)).all())
                self.assertTrue((curves[:, 1:] <= curves[:, :-1] + 1e-6).all())
                self.assertTrue((curves[:, 0] > 0.99).all())

    def test_density_is_negative_survival_derivative(self):
        # p(t) = -dS/dt, which checks the Jacobian and time-rescaling terms.
        for config in self.configs:
            with self.subTest(**config):
                model = self._model(**config)
                params = model.mixture_params(self.batch["features"])
                params = tuple(p.detach() for p in params)
                t = torch.linspace(0.5, 40.0, 64).requires_grad_(True)
                surv = model.log_survival(params, t).exp()
                (dsdt,) = torch.autograd.grad(surv.sum(), t)
                density = model.log_density(params, t).exp()
                torch.testing.assert_close(density, -dsdt, rtol=1e-3, atol=1e-6)

    def test_matches_closed_form_densities(self):
        mu = torch.tensor([[0.3], [-0.5]])
        sigma = torch.tensor([[0.8], [1.2]])
        params = (torch.zeros(2, 1), mu, sigma)
        t = torch.tensor([2.0, 0.7])
        scale = 3.0
        u = t / scale
        normal = torch.distributions.Normal(mu.squeeze(1), sigma.squeeze(1))
        # exp transform: log-normal density in original units.
        model = self._model(transform="exp", num_components=1, time_scale=scale)
        y = torch.log(u)
        expected = normal.log_prob(y) - torch.log(t)
        torch.testing.assert_close(model.log_density(params, t), expected)
        expected_s = torch.log1p(-normal.cdf(y))
        torch.testing.assert_close(model.log_survival(params, t), expected_s)
        # softplus transform: y = log(e^u - 1), dy/dt = e^u / (e^u - 1) / s.
        model = self._model(transform="softplus", num_components=1, time_scale=scale)
        y = torch.log(torch.expm1(u))
        log_jac = u - torch.log(torch.expm1(u)) - torch.log(torch.tensor(scale))
        expected = normal.log_prob(y) + log_jac
        torch.testing.assert_close(model.log_density(params, t), expected)

    def test_single_component_starts_at_zero(self):
        model = self._model(num_components=1)
        self.assertEqual(model.mean_head.bias.detach().item(), 0.0)
        model = self._model(num_components=4)
        torch.testing.assert_close(
            model.mean_head.bias.detach(), torch.linspace(-3.0, 3.0, 4)
        )

    def test_reference_architecture(self):
        model = self._model(num_components=4)
        layers = list(model.encoder)
        self.assertEqual(len(layers), 9)
        for i in range(3):
            linear, norm, act = layers[3 * i : 3 * i + 3]
            self.assertIsInstance(linear, torch.nn.Linear)
            self.assertIsNone(linear.bias)
            self.assertEqual(linear.out_features, 32)
            self.assertIsInstance(norm, torch.nn.BatchNorm1d)
            self.assertIsInstance(act, torch.nn.PReLU)
        # "fixed" init: uniform weights and unit scales for every patient.
        log_w, _, scales = model.mixture_params(self.batch["features"])
        torch.testing.assert_close(log_w.exp(), torch.full((64, 4), 0.25))
        torch.testing.assert_close(scales, torch.ones(64, 4))
        plain = self._model(batch_norm=False, init="default", num_layers=2)
        self.assertFalse(
            any(isinstance(m, torch.nn.BatchNorm1d) for m in plain.encoder)
        )
        self.assertIsNotNone(plain.encoder[0].bias)
        # Without BatchNorm a single-patient training batch works.
        one = {k: v[:1] for k, v in self.batch.items() if k != "patient_id"}
        self.assertTrue(torch.isfinite(plain(**one)["loss"]))
        with self.assertRaises(ValueError):
            self._model(init="residual")

    def test_checkpoint_carries_time_scale(self):
        trained = self._model(num_components=3, time_scale=7.0)
        self.assertIn("time_scale", trained.state_dict())
        rebuilt = self._model(num_components=3, time_scale=1.0)
        rebuilt.load_state_dict(trained.state_dict())
        self.assertEqual(float(rebuilt.time_scale), 7.0)
        grid = torch.tensor([1.0, 5.0, 20.0])
        features = self.batch["features"][:4]
        torch.testing.assert_close(
            rebuilt.predict_survival(features, grid),
            trained.predict_survival(features, grid),
        )

    def test_predict_survival_uses_eval_mode(self):
        model = self._model(num_components=3)
        model.train()
        grid = torch.tensor([0.0, 1.0, 10.0])
        curves = model.predict_survival(self.batch["features"][:5], grid)
        self.assertTrue(model.training)
        torch.testing.assert_close(curves[:, 0], torch.ones(5))
        model.eval()
        torch.testing.assert_close(
            model.predict_survival(self.batch["features"][:5], grid), curves
        )
        with self.assertRaises(ValueError):
            model.predict_survival(self.batch["features"][:5], torch.tensor([-1.0]))

    def test_rejects_invalid_labels(self):
        model = self._model()
        params = model.mixture_params(self.batch["features"][:2])
        bad = {
            "zero_time": [[0.0, 1.0], [1.0, 0.0]],
            "negative_time": [[-1.0, 1.0], [1.0, 0.0]],
            "nan_time": [[float("nan"), 1.0], [1.0, 0.0]],
            "bad_event": [[1.0, 2.0], [1.0, 0.0]],
        }
        for name, labels in bad.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                model.censored_nll(params, torch.tensor(labels))
        with self.assertRaises(ValueError):
            model.censored_nll(params, torch.ones(2, 3))


class TestSurvivalMDNTraining(unittest.TestCase):
    """End-to-end training on synthetic data."""

    def test_recovers_true_survival_curve(self):
        torch.manual_seed(0)
        samples = make_survival_samples(n=4000, censoring_rate=0.3, seed=1)
        dataset = _dataset(samples, name="recovery")
        batch = _batch(dataset, batch_size=4000)
        # A small network avoids overfitting the 4,000 training patients.
        model = SurvivalMDN(
            dataset, num_components=3, hidden_dim=16, num_layers=1, time_scale=7.0
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(400):
            optimizer.zero_grad()
            model(**batch)["loss"].backward()
            optimizer.step()
        model.eval()
        test = make_survival_samples(n=300, censoring_rate=0.0, seed=2)
        features = torch.tensor([s["features"] for s in test])
        t_grid = torch.linspace(1.0, 30.0, 30)
        predicted = model.predict_survival(features, t_grid).numpy()
        truth = true_survival(features.numpy(), t_grid.numpy())
        self.assertLess(float(np.abs(predicted - truth).mean()), 0.03)

    def test_trainer_checkpoint_round_trip(self):
        import tempfile

        from pyhealth.trainer import Trainer

        torch.manual_seed(0)
        train = _dataset(make_survival_samples(n=96, seed=3), name="train")
        val = _dataset(make_survival_samples(n=32, seed=4), name="val")
        model = SurvivalMDN(train, num_components=3, time_scale=7.0)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(model, device="cpu", output_path=tmp, exp_name="t")
            trainer.train(
                get_dataloader(train, batch_size=32, shuffle=True),
                get_dataloader(val, batch_size=32),
                epochs=3,
                monitor="loss",
                monitor_criterion="min",
            )
            scores = trainer.evaluate(get_dataloader(val, batch_size=32))
            self.assertEqual(list(scores), ["loss"])
            self.assertTrue(np.isfinite(scores["loss"]))

            features = _batch(val)["features"]
            t_grid = torch.linspace(1.0, 30.0, 10)
            model.eval()
            before = model.predict_survival(features, t_grid)
            reloaded = SurvivalMDN(train, num_components=3, time_scale=7.0)
            Trainer(
                reloaded,
                checkpoint_path=f"{tmp}/t/best.ckpt",
                device="cpu",
                enable_logging=False,
            )
            reloaded.eval()
            after = reloaded.predict_survival(features, t_grid)
            torch.testing.assert_close(before, after)


if __name__ == "__main__":
    unittest.main()
