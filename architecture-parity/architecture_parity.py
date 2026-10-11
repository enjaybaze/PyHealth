"""Transferred-weight parity: NJ's SurvivalMDN vs the authors' released model.

Builds the authors' ``MukundMDNHelper`` (``init_type="fixed"``), randomises all
of its weights and BatchNorm statistics, copies them into
``pyhealth.models.SurvivalMDN``, and compares, in training mode (batch
statistics) and evaluation mode (running statistics):

- mixture parameters: weights, means and scales;
- log density and survival at observed-style times, against the authors'
  ``SurvivalMDN_Dist``.

The authors' code is imported at run time only; none of it is in the PR.
Their density adds small epsilons (1e-6 inside the inverse softplus and the
log-sum-exp), so densities are compared on moderate times with a tolerance.

Usage:
    python -I architecture_parity.py <authors' repo> <pyhealth repo> [output json]
"""

import json
import sys
from pathlib import Path

import torch

ref_repo, pyhealth_repo = Path(sys.argv[1]), Path(sys.argv[2])
out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
sys.path.insert(0, str(pyhealth_repo))
sys.path.insert(1, str(ref_repo))

from mdn_models import MukundMDNHelper, SurvivalMDN_Dist  # noqa: E402

from pyhealth.datasets import create_sample_dataset  # noqa: E402
from pyhealth.models import SurvivalMDN  # noqa: E402


def build_pair(d_in, hidden, k, seed):
    torch.manual_seed(seed)
    theirs = MukundMDNHelper(d_in, hidden, "fixed", k)
    samples = [
        {"patient_id": f"p{i}", "features": [0.0] * d_in, "survival": [1.0, 1.0]}
        for i in range(2)
    ]
    ds = create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"survival": "tensor"},
        dataset_name="parity",
    )
    ours = SurvivalMDN(ds, num_components=k, hidden_dim=hidden, num_layers=3)
    with torch.no_grad():
        for p in theirs.parameters():
            p.copy_(0.5 * torch.randn_like(p))
        for m in theirs.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.copy_(0.3 * torch.randn_like(m.running_mean))
                m.running_var.copy_(torch.rand_like(m.running_var) + 0.5)
        src = list(theirs.fc_in.net)
        dst = list(ours.encoder)
        assert len(src) == len(dst) == 9
        for a, b in zip(src, dst):
            assert type(a) is type(b), (type(a), type(b))
            b.load_state_dict(a.state_dict())
        ours.weight_head.load_state_dict(theirs.pi_transform[0].state_dict())
        ours.mean_head.load_state_dict(theirs.mu_transform[0].state_dict())
        ours.scale_head.load_state_dict(theirs.sigma_transform[0].state_dict())
    return theirs, ours


def compare(theirs, ours, x, times, training):
    theirs.train(training)
    ours.train(training)
    with torch.no_grad():
        pred = theirs(x)
        dist = SurvivalMDN_Dist(pred)
        w_ref, mu_ref, sd_ref = dist.pred_params_to_parameters(pred)
        log_w, mu, sd = ours.mixture_params(x)
        params = (log_w, mu, sd)
        logp = ours.log_density(params, times)
        surv = ours.log_survival(params, times).exp()
        logp_ref = dist.log_prob(times)
        surv_ref = 1.0 - dist.cdf(times)
        # The reference adds 1e-6 inside its log-sum-exp after factoring out
        # the largest term. Where the remaining sum is small that epsilon
        # dominates and floors the density, so compare only where the sum
        # is at least 1e-2 (epsilon contributes < 1e-4 in log space there).
        y = torch.log(torch.expm1(times) + 1e-6).unsqueeze(-1)
        comp = torch.distributions.Normal(mu_ref, sd_ref).log_prob(y)
        top = comp.max(dim=-1, keepdim=True).values
        rest = (w_ref * torch.exp(comp - top)).sum(-1)
        unfloored = rest >= 1e-2
        floored_diff = logp_ref - logp
    return {
        "weights_max_abs": float((log_w.exp() - w_ref).abs().max()),
        "means_max_abs": float((mu - mu_ref).abs().max()),
        "scales_max_abs": float((sd - sd_ref).abs().max()),
        "log_density_max_abs": float((logp - logp_ref)[unfloored].abs().max()),
        "log_density_points_compared": int(unfloored.sum()),
        "floored_points_reference_higher": bool((floored_diff[~unfloored] > -1e-4).all()),
        "survival_max_abs": float((surv - surv_ref).abs().max()),
    }


def main():
    results = []
    for d_in, hidden, k, seed in [(7, 32, 5, 0), (27, 64, 10, 1), (7, 8, 1, 2), (9, 128, 20, 3)]:
        theirs, ours = build_pair(d_in, hidden, k, seed)
        torch.manual_seed(100 + seed)
        x = torch.randn(256, d_in)
        times = torch.rand(256) * 4.0 + 0.25  # moderate times in model units
        for training in (True, False):
            r = compare(theirs, ours, x, times, training)
            r.update(d_in=d_in, hidden=hidden, k=k, mode="train" if training else "eval")
            results.append(r)
            print(
                f"D={d_in:2d} hidden={hidden:3d} K={k:2d} {r['mode']:5s} | "
                f"weights {r['weights_max_abs']:.1e} means {r['means_max_abs']:.1e} "
                f"scales {r['scales_max_abs']:.1e} | log p {r['log_density_max_abs']:.1e} "
                f"({r['log_density_points_compared']}/256 unfloored) "
                f"S(t) {r['survival_max_abs']:.1e}"
            )
    params_ok = all(
        max(r["weights_max_abs"], r["means_max_abs"], r["scales_max_abs"]) < 1e-5
        for r in results
    )
    dist_ok = all(
        r["log_density_max_abs"] < 1e-3
        and r["survival_max_abs"] < 1e-5
        and r["floored_points_reference_higher"]
        for r in results
    )
    print("mixture parameters identical:", params_ok)
    print("density/survival within epsilon tolerance:", dist_ok)
    if out_path:
        out_path.write_text(json.dumps({"params_ok": params_ok, "dist_ok": dist_ok,
                                        "runs": results}, indent=2) + "\n")
    assert params_ok and dist_ok


if __name__ == "__main__":
    main()
