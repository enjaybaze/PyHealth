# Released GBSG splits (T3, Oct 18, 2026)

NJ took over this handoff task (t3-05) while Aritra is away.
Code: `feat/survival-mdn` commit `198659d`.

## What was added to the PR

- `pyhealth.datasets.gbsg_released_split_ids(root, released_dir, repeat)`:
  maps each row of the authors' `{train,valid,test}_{repeat}.npz` to a
  `source_id`, matching features, event and time (`months / max + 0.001`,
  tolerance 1e-6). Fails if a row has no unused match or a source row is
  left over.
- `pyhealth.datasets.gbsg_paper_time_transform(root)`: returns
  `{"time_divisor": 87.35934448242188, "time_offset": 0.002}` for
  `SurvivalTimeToEventGBSG`. The extra 0.001 is the authors' loader
  (`t = y[:, 0] + 0.001`).
- `SurvivalTimeToEventGBSG(time_divisor=..., time_offset=...)`.
- Example: `--released-dir DIR --repeat N` runs the paper protocol (raw
  features, `time_scale=1.0`) and reports test C^td / IBS / IBLL.

## Verification (`verify_released_splits.py`, `results.json`)

| Check | Result |
| --- | --- |
| Sizes, every repeat | 1785 / 223 / 224 |
| One-to-one over 2,232 source rows | all 10 repeats |
| Max abs error, mapped row vs released row | 4.8e-8 |
| Agreement with Aritra's data-audit manifest | 22,280 of 22,300 rows; the other 20 differ only in which identical twin is named |

The two twin groups are `(train_391, train_392)` and `(train_834,
train_1015)`: identical in every recorded value, so the choice cannot change
any split's data. Aritra's manifest names the same twin for both released
rows in each pair; ours assigns each source row once.

Example run, repeat 1, 5 epochs, CPU: validation NLL 0.6251 → 0.3844,
checkpoint reload error 0, test C^td 0.639, IBS 0.186, IBLL -0.544. On
repeat 1 the test set's censoring KM never drops below 0.57, so all three
horizons (1e-8, 0.2, 0.4) coincide. These five-epoch numbers are a smoke
check, not results.
