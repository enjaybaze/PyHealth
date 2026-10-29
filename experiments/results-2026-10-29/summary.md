# Course run results (300 fits)

Selection on validation only: MDN by validation NLL among its three tuning candidates, baselines by validation IBS among their three configs, per split. Test means ± SD over the 10 released splits. "ref." = the authors' widest-horizon convention, used for the paper comparison.

| Dataset | Model | C^td | IBS (aligned) | IBLL (aligned) | IBS (ref.) | IBLL (ref.) | Paper C^td / IBS / IBLL | D-cal p > 0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GBSG | cox | 0.658 ± 0.019 | 0.174 ± 0.010 | −0.515 ± 0.026 | 0.174 | −0.514 | 0.645 / 0.177 / −0.523 | 10/10 |
| GBSG | deepsurv | 0.675 ± 0.016 | 0.167 ± 0.011 | −0.497 ± 0.030 | 0.167 | −0.497 | 0.663 / 0.172 / −0.509 | 10/10 |
| GBSG | mdn | 0.663 ± 0.020 | 0.173 ± 0.008 | −0.513 ± 0.022 | 0.173 | −0.512 | 0.668 / 0.172 / −0.504 | 9/10 |
| SUPPORT | cox | 0.596 ± 0.009 | 0.194 ± 0.002 | −0.569 ± 0.005 | 0.194 | −0.569 | 0.596 / 0.194 / −0.568 | 6/10 |
| SUPPORT | deepsurv | 0.604 ± 0.008 | 0.191 ± 0.003 | −0.563 ± 0.009 | 0.191 | −0.563 | 0.609 / 0.190 / −0.559 | 7/10 |
| SUPPORT | mdn | 0.626 ± 0.010 | 0.189 ± 0.002 | −0.559 ± 0.007 | 0.189 | −0.559 | 0.628 / 0.190 / −0.559 | 8/10 |

## MDN minus baselines (test, paired by split)

| Dataset | Comparison | Metric | Mean diff | SD | Splits MDN better | Paired t p |
| --- | --- | --- | --- | --- | --- | --- |
| GBSG | mdn - deepsurv | c_td | −0.0127 | 0.0167 | 2/10 | 0.039 |
| GBSG | mdn - deepsurv | ibs | 0.0061 | 0.0063 | 2/10 | 0.014 |
| GBSG | mdn - deepsurv | ibll | −0.0157 | 0.0161 | 2/10 | 0.013 |
| GBSG | mdn - cox | c_td | 0.0051 | 0.0217 | 9/10 | 0.478 |
| GBSG | mdn - cox | ibs | −0.0008 | 0.0061 | 5/10 | 0.690 |
| GBSG | mdn - cox | ibll | 0.0017 | 0.0145 | 5/10 | 0.716 |
| SUPPORT | mdn - deepsurv | c_td | 0.0219 | 0.0091 | 10/10 | 0.000 |
| SUPPORT | mdn - deepsurv | ibs | −0.0021 | 0.0017 | 8/10 | 0.004 |
| SUPPORT | mdn - deepsurv | ibll | 0.0042 | 0.0044 | 8/10 | 0.014 |
| SUPPORT | mdn - cox | c_td | 0.0306 | 0.0108 | 10/10 | 0.000 |
| SUPPORT | mdn - cox | ibs | −0.0047 | 0.0010 | 10/10 | 0.000 |
| SUPPORT | mdn - cox | ibll | 0.0096 | 0.0040 | 10/10 | 0.000 |

## Mixture size K (softplus, default settings)

| Dataset | K | Val NLL | C^td | IBS | IBLL | Δ val NLL vs K=10 (p) |
| --- | --- | --- | --- | --- | --- | --- |
| GBSG | 1 | 0.3929 | 0.664 | 0.175 | −0.517 | 0.0265 (0.009) |
| GBSG | 2 | 0.3713 | 0.651 | 0.174 | −0.515 | 0.0049 (0.339) |
| GBSG | 3 | 0.3647 | 0.663 | 0.173 | −0.513 | −0.0017 (0.591) |
| GBSG | 5 | 0.3636 | 0.662 | 0.173 | −0.514 | −0.0028 (0.184) |
| GBSG | 10 | 0.3664 | 0.662 | 0.175 | −0.516 | 0.0000 (1.000) |
| GBSG | 20 | 0.3647 | 0.660 | 0.175 | −0.517 | −0.0017 (0.356) |
| SUPPORT | 1 | 0.6944 | 0.623 | 0.193 | −0.578 | 0.1625 (0.000) |
| SUPPORT | 2 | 0.5733 | 0.625 | 0.190 | −0.559 | 0.0414 (0.000) |
| SUPPORT | 3 | 0.5333 | 0.626 | 0.189 | −0.557 | 0.0015 (0.316) |
| SUPPORT | 5 | 0.5313 | 0.626 | 0.189 | −0.558 | −0.0006 (0.451) |
| SUPPORT | 10 | 0.5319 | 0.627 | 0.189 | −0.559 | 0.0000 (1.000) |
| SUPPORT | 20 | 0.5329 | 0.626 | 0.189 | −0.558 | 0.0011 (0.109) |

## Transform (K=10): softplus minus exp, paired by split

| Dataset | Metric | Softplus | Exp | Diff | Splits softplus better | p |
| --- | --- | --- | --- | --- | --- | --- |
| GBSG | val_nll | 0.3664 | 0.3620 | 0.0044 | 3/10 | 0.102 |
| GBSG | c_td | 0.6616 | 0.6621 | −0.0005 | 4/10 | 0.868 |
| GBSG | ibs | 0.1746 | 0.1728 | 0.0018 | 4/10 | 0.208 |
| GBSG | ibll | −0.5161 | −0.5117 | −0.0045 | 4/10 | 0.140 |
| SUPPORT | val_nll | 0.5319 | 0.5171 | 0.0148 | 0/10 | 0.000 |
| SUPPORT | c_td | 0.6267 | 0.6256 | 0.0011 | 5/10 | 0.498 |
| SUPPORT | ibs | 0.1893 | 0.1897 | −0.0005 | 8/10 | 0.120 |
| SUPPORT | ibll | −0.5593 | −0.5581 | −0.0012 | 4/10 | 0.197 |

## Tuning candidates: wins by split (validation only)

| Dataset | Model | Config | Wins | Mean validation score |
| --- | --- | --- | --- | --- |
| GBSG | mdn | mdn_k10_softplus | 3/10 | 0.3664 |
| GBSG | mdn | mdn_slowlr | 1/10 | 0.3694 |
| GBSG | mdn | mdn_wide | 6/10 | 0.3636 |
| GBSG | cox | cox_a0.01 | 3/10 | 0.1737 |
| GBSG | cox | cox_a0.1 | 0/10 | 0.1737 |
| GBSG | cox | cox_a1 | 7/10 | 0.1737 |
| GBSG | deepsurv | deepsurv_base | 6/10 | 0.1695 |
| GBSG | deepsurv | deepsurv_slowlr | 0/10 | 0.1769 |
| GBSG | deepsurv | deepsurv_wide | 4/10 | 0.1694 |
| SUPPORT | mdn | mdn_k10_softplus | 4/10 | 0.5319 |
| SUPPORT | mdn | mdn_slowlr | 3/10 | 0.5339 |
| SUPPORT | mdn | mdn_wide | 3/10 | 0.5329 |
| SUPPORT | cox | cox_a0.01 | 7/10 | 0.1942 |
| SUPPORT | cox | cox_a0.1 | 0/10 | 0.1942 |
| SUPPORT | cox | cox_a1 | 3/10 | 0.1942 |
| SUPPORT | deepsurv | deepsurv_base | 1/10 | 0.1950 |
| SUPPORT | deepsurv | deepsurv_slowlr | 0/10 | 0.2071 |
| SUPPORT | deepsurv | deepsurv_wide | 9/10 | 0.1935 |
