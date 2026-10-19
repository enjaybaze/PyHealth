"""Survival Mixture Density Network (Han, Goldstein and Ranganath, 2022).

Reference:
    X. Han, M. Goldstein, and R. Ranganath. Survival Mixture Density
    Networks. Proceedings of the 7th Machine Learning for Healthcare
    Conference, PMLR 182:224-248, 2022.

The authors' public code was used only as a reference for the equations;
this implementation is written independently against PyHealth's APIs.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

VALID_TRANSFORMS = ("softplus", "exp")
VALID_INITS = ("fixed", "default")

# Reference initialisation and output clamps (the authors' released code).
# Initial latent means are spread evenly over this range when K > 1. A single
# component starts at 0 (the K = 1 control), not at an endpoint of the range.
_INIT_MEAN_RANGE = (-3.0, 3.0)
_HIDDEN_CLAMP = 1e3  # on the encoder output before the weight head
_MEAN_CLAMP = 30.0
_RAW_SCALE_CLAMP = (1e-2, 1e2)  # before softplus, so scales >= softplus(0.01)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

Params = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class SurvivalMDN(BaseModel):
    """Survival Mixture Density Network for right-censored time-to-event data.

    An MLP maps patient covariates ``x`` to a ``K``-component Gaussian
    mixture over a latent value ``y``. A monotone transform ``t = g(y)``
    maps the latent value to a positive event time: softplus (the paper's
    choice) or exp (a log-normal mixture). Because ``g`` is invertible, the
    event-time density ``p(t | x)`` and survival function ``S(t | x)`` have
    closed forms, so the model trains on the exact censored likelihood with
    minibatch SGD.

    The network follows the authors' released ``default``/``fixed``
    architecture: ``num_layers`` bias-free ``Linear -> BatchNorm1d -> PReLU``
    blocks, then three linear heads for the mixture weights (softmax),
    means (clamped to [-30, 30]) and scales (``softplus`` of the head
    output clamped to [0.01, 100]). The likelihood is computed exactly in
    log space; the reference code's small epsilon corrections and hazard
    floor are deliberately not reproduced. With ``batch_norm=True`` a
    training batch needs at least two patients.

    Data contract (agreed in T1):

    - Input: one feature key holding a float tensor of shape ``[B, D]``.
    - Label: one label key holding a float tensor of shape ``[B, 2]``,
      column 0 the observed time (event or censoring time) and column 1
      the event indicator (1 = event observed, 0 = right censored).
    - The task declares the label as ``"tensor"``, so ``model.mode`` is
      ``None`` and :class:`~pyhealth.trainer.Trainer` trains on the loss
      returned by :meth:`forward`. Train with ``monitor="loss"`` and
      ``monitor_criterion="min"``.
    - Times must be finite and strictly positive. Invalid labels raise a
      ``ValueError`` instead of being repaired.

    Args:
        dataset: the processed dataset; used to read the feature and label
            keys and the input dimension.
        num_components: number of mixture components ``K``. Default 10.
        transform: positive-time transform, ``"softplus"`` or ``"exp"``.
            Default ``"softplus"``.
        hidden_dim: width of every hidden layer. Default 32. The authors
            tuned it between 8 and 128.
        num_layers: hidden blocks in the shared encoder, at least 1.
            Default 3, as in the reference.
        batch_norm: use the reference bias-free ``Linear -> BatchNorm1d ->
            PReLU`` blocks. ``False`` uses ``Linear -> PReLU`` with biases,
            which also works with a training batch of one. Default True.
        init: head initialisation, ``"fixed"`` (uniform weights, unit
            scales, means evenly spaced on [-3, 3]) or ``"default"``
            (PyTorch's initialisation for the weight and scale heads, the
            same mean initialisation). A single component always starts
            at mean 0. Default ``"fixed"``.
        dropout: dropout rate after each hidden block (not in the
            reference). Default 0.0.
        time_scale: times are divided by this value inside the model, and
            the matching Jacobian term is added back, so the loss and
            densities stay in the dataset's original time units. Set it
            to a training-set statistic (for example the median observed
            time). Stored as a buffer, so saved checkpoints carry it and
            loading one restores the value it was trained with. Default 1.0.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "features": [0.2, 1.0, 0.4],
        ...      "survival": [12.5, 1.0]},
        ...     {"patient_id": "p1", "features": [0.7, 0.0, 1.3],
        ...      "survival": [30.0, 0.0]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"survival": "tensor"},
        ...     dataset_name="toy_survival",
        ... )
        >>> model = SurvivalMDN(dataset, num_components=4, time_scale=20.0)
        >>> model.mode is None
        True
        >>> from pyhealth.datasets import get_dataloader
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> out = model(**batch)
        >>> out["loss"].dim()
        0
        >>> t_grid = torch.linspace(1.0, 40.0, steps=5)
        >>> model.predict_survival(batch["features"], t_grid).shape
        torch.Size([2, 5])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        num_components: int = 10,
        transform: str = "softplus",
        hidden_dim: int = 32,
        num_layers: int = 3,
        batch_norm: bool = True,
        init: str = "fixed",
        dropout: float = 0.0,
        time_scale: float = 1.0,
    ) -> None:
        super().__init__(dataset)
        if transform not in VALID_TRANSFORMS:
            raise ValueError(
                f"transform must be one of {VALID_TRANSFORMS}, got {transform!r}"
            )
        if init not in VALID_INITS:
            raise ValueError(f"init must be one of {VALID_INITS}, got {init!r}")
        if num_components < 1:
            raise ValueError("num_components must be at least 1")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if len(self.feature_keys) != 1 or len(self.label_keys) != 1:
            raise ValueError(
                "SurvivalMDN expects exactly one feature key and one label key"
            )
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.num_components = num_components
        self.transform = transform
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.init = init
        self.dropout = dropout
        # A buffer, not a plain float, so it is saved in checkpoints.
        self.register_buffer(
            "time_scale", torch.tensor(float(time_scale), dtype=torch.float32)
        )

        input_dim = int(torch.as_tensor(dataset[0][self.feature_key]).numel())
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(width, hidden_dim, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.PReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.weight_head = nn.Linear(hidden_dim, num_components)
        self.mean_head = nn.Linear(hidden_dim, num_components)
        self.scale_head = nn.Linear(hidden_dim, num_components)
        self._init_heads()

    def _init_heads(self) -> None:
        """Applies the reference head initialisation."""
        k = self.num_components
        with torch.no_grad():
            means = torch.zeros(1) if k == 1 else torch.linspace(*_INIT_MEAN_RANGE, k)
            self.mean_head.weight.zero_()
            self.mean_head.bias.copy_(means)
            if self.init == "fixed":
                self.weight_head.weight.zero_()
                self.weight_head.bias.fill_(1.0 / k)
                # softplus(log(e - 1)) = 1: unit scales.
                self.scale_head.weight.zero_()
                self.scale_head.bias.fill_(math.log(math.e - 1.0))

    def _latent(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Maps original-unit times to latent values ``y = g^{-1}(t / s)``.

        Returns ``(y, log_jacobian)`` where ``log_jacobian`` is
        ``log |dy/dt|``, including the ``-log s`` time-rescaling term.
        """
        u = times / self.time_scale
        if self.transform == "softplus":
            # g^{-1}(u) = log(e^u - 1) = u + log(1 - e^{-u}), stable for all u > 0.
            log1mexp = torch.log(-torch.expm1(-u))
            y = u + log1mexp
            log_jac = -log1mexp
        else:
            y = torch.log(u)
            log_jac = -y
        return y, log_jac - torch.log(self.time_scale)

    @staticmethod
    def _align(params: Params, times: torch.Tensor) -> tuple[Params, torch.Tensor]:
        """Broadcasts ``[B, K]`` params against ``[B]`` or ``[B, T]`` times."""
        if times.dim() == 1:
            return params, times
        if times.dim() == 2:
            return tuple(p.unsqueeze(1) for p in params), times
        raise ValueError("times must have shape [B] or [B, T]")

    @staticmethod
    def _check_times(times: torch.Tensor) -> None:
        if not torch.isfinite(times).all() or (times <= 0).any():
            raise ValueError("survival times must be finite and strictly positive")

    def mixture_params(self, features: torch.Tensor) -> Params:
        """Predicts the latent Gaussian mixture for each patient.

        Args:
            features: float tensor of shape ``[B, D]``.

        Returns:
            ``(log_weights, means, scales)``, each of shape ``[B, K]``.
            ``log_weights`` are log-softmax normalised; ``means`` lie in
            [-30, 30]; ``scales`` are latent standard deviations of at
            least ``softplus(0.01)`` (about 0.70).

        Examples:
            >>> log_w, mu, sigma = model.mixture_params(batch["features"])
            >>> log_w.shape == mu.shape == sigma.shape
            True
        """
        h = self.encoder(features.to(self.device).float())
        logits = self.weight_head(h.clamp(-_HIDDEN_CLAMP, _HIDDEN_CLAMP))
        log_weights = F.log_softmax(logits, dim=-1)
        means = self.mean_head(h).clamp(-_MEAN_CLAMP, _MEAN_CLAMP)
        scales = F.softplus(self.scale_head(h).clamp(*_RAW_SCALE_CLAMP))
        return log_weights, means, scales

    def log_density(self, params: Params, times: torch.Tensor) -> torch.Tensor:
        """Log event-time density ``log p(t | x)`` in original time units.

        Args:
            params: output of :meth:`mixture_params`, each ``[B, K]``.
            times: positive times of shape ``[B]`` (one per patient) or
                ``[B, T]`` (a grid per patient).

        Returns:
            Tensor with the same shape as ``times``.

        Examples:
            >>> params = model.mixture_params(batch["features"])
            >>> model.log_density(params, batch["survival"][:, 0]).shape
            torch.Size([2])
        """
        times = times.to(self.device).float()
        self._check_times(times)
        (log_w, mu, sigma), times = self._align(params, times)
        y, log_jac = self._latent(times)
        z = (y.unsqueeze(-1) - mu) / sigma
        log_normal = -0.5 * z**2 - torch.log(sigma) - _LOG_SQRT_2PI
        return torch.logsumexp(log_w + log_normal, dim=-1) + log_jac

    def log_survival(self, params: Params, times: torch.Tensor) -> torch.Tensor:
        """Log survival probability ``log S(t | x)``, computed stably.

        Args:
            params: output of :meth:`mixture_params`, each ``[B, K]``.
            times: positive times of shape ``[B]`` or ``[B, T]``.

        Returns:
            Tensor with the same shape as ``times``.

        Examples:
            >>> params = model.mixture_params(batch["features"])
            >>> model.log_survival(params, batch["survival"][:, 0]).shape
            torch.Size([2])
        """
        times = times.to(self.device).float()
        self._check_times(times)
        (log_w, mu, sigma), times = self._align(params, times)
        y, _ = self._latent(times)
        z = (y.unsqueeze(-1) - mu) / sigma
        # S(t) = sum_k w_k * (1 - Phi(z_k)) = sum_k w_k * Phi(-z_k).
        return torch.logsumexp(log_w + torch.special.log_ndtr(-z), dim=-1)

    def censored_nll(self, params: Params, survival: torch.Tensor) -> torch.Tensor:
        """Mean censored negative log-likelihood over the batch.

        Observed events contribute ``-log p(t | x)``; censored patients
        contribute ``-log S(t | x)``.

        Args:
            params: output of :meth:`mixture_params`, each ``[B, K]``.
            survival: label tensor of shape ``[B, 2]`` holding
                ``[observed time, event indicator]``.

        Returns:
            Scalar tensor.

        Raises:
            ValueError: if any time is not finite and positive, or any
                event indicator is not 0 or 1.

        Examples:
            >>> params = model.mixture_params(batch["features"])
            >>> model.censored_nll(params, batch["survival"]).dim()
            0
        """
        survival = survival.to(self.device).float()
        if survival.dim() != 2 or survival.shape[1] != 2:
            raise ValueError("survival labels must have shape [B, 2]")
        times, events = survival[:, 0], survival[:, 1]
        if not ((events == 0) | (events == 1)).all():
            raise ValueError("event indicators must be 0 or 1")
        log_p = self.log_density(params, times)
        log_s = self.log_survival(params, times)
        return -(events * log_p + (1.0 - events) * log_s).mean()

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Computes the training loss for a batch.

        Args:
            **kwargs: a batch from :func:`~pyhealth.datasets.get_dataloader`.
                Must contain the feature key (``[B, D]``) and the label key
                (``[B, 2]``).

        Returns:
            A dictionary with:

            - ``loss``: scalar mean censored negative log-likelihood.
            - ``y_true``: the ``[B, 2]`` label tensor, in batch row order.
            - ``log_weights``, ``means``, ``scales``: the mixture
              parameters, each ``[B, K]``.

        Examples:
            >>> out = model(**batch)
            >>> sorted(out)
            ['log_weights', 'loss', 'means', 'scales', 'y_true']
        """
        params = self.mixture_params(kwargs[self.feature_key])
        y_true = kwargs[self.label_key].to(self.device).float()
        loss = self.censored_nll(params, y_true)
        log_weights, means, scales = params
        return {
            "loss": loss,
            "y_true": y_true,
            "log_weights": log_weights,
            "means": means,
            "scales": scales,
        }

    @torch.no_grad()
    def predict_survival(
        self, features: torch.Tensor, t_grid: torch.Tensor
    ) -> torch.Tensor:
        """Predicts survival curves ``S(t | x)`` on a shared time grid.

        Rows follow the order of ``features``; columns follow ``t_grid``.
        ``t_grid`` is in the dataset's original time units. Predictions
        always use evaluation mode (BatchNorm running statistics, no
        dropout); the model's previous mode is restored afterwards.

        Args:
            features: float tensor of shape ``[B, D]``.
            t_grid: 1-D tensor of ``T`` non-negative, increasing times.
                ``S(0) = 1``.

        Returns:
            Tensor of shape ``[B, T]`` with values in ``[0, 1]``,
            non-increasing along each row.

        Examples:
            >>> t_grid = torch.linspace(0.0, 40.0, steps=5)
            >>> model.predict_survival(batch["features"], t_grid).shape
            torch.Size([2, 5])
        """
        t_grid = torch.as_tensor(t_grid, dtype=torch.float32, device=self.device)
        if t_grid.dim() != 1:
            raise ValueError("t_grid must be a 1-D tensor")
        if not torch.isfinite(t_grid).all() or (t_grid < 0).any():
            raise ValueError("t_grid must be finite and non-negative")
        was_training = self.training
        self.eval()
        try:
            params = self.mixture_params(features)
        finally:
            self.train(was_training)
        positive = t_grid > 0
        times = torch.where(positive, t_grid, torch.ones_like(t_grid))
        times = times.unsqueeze(0).expand(params[0].shape[0], -1)
        surv = self.log_survival(params, times).exp().clamp(0.0, 1.0)
        return torch.where(positive, surv, torch.ones_like(surv))
