# Vice Trophy Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![make.fun rank · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ocoyan31-sys/vice-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![make.fun reward · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ocoyan31-sys/vice-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for the make.fun competition **gta6-trophy-rarity-forecast-01**:
for every PlayStation 5 trophy of Grand Theft Auto VI, it predicts the
earn rate printed on Exophase at the case's target instant — the percent
of tracked players who have earned the trophy — as five quantiles
(5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## How it works

`main.py` is a single-file Python 3 program using only the standard
library. It reads every `*.in` case file in `$MAKEFUN_INPUT_DIR`
(default `/input`) in name order and writes one `<id>.out` JSON file per
case to `$MAKEFUN_OUTPUT_DIR` (default `/output`). It is deterministic:
two runs on the same case are byte-identical, and it runs in a small
fraction of the 8-second per-case limit.

### Epoch collapse

Exophase refreshes a game's counts on its own schedule, so the recorded
history is long runs of identical reads punctuated by jumps. Consecutive
identical reads are collapsed into distinct refresh epochs before any
fitting happens; fitting the raw reads would overweight the frozen
stretches.

### Earners and players, modelled separately

A rate is earners / players_tracked, and the two parts behave very
differently at a refresh: the denominator can multiply in one step —
diluting almost every rate even while earners rise — while earners only
ever grow. The forecaster therefore projects them separately:

- **Players** are projected with a recency-weighted fit against
  log-time, bending the launch surge into slower later growth, clamped
  to never fall below the last count. With only one flat epoch, a
  generic growth factor rising with the square root of the horizon is
  used instead.
- **Earners** per trophy are rebuilt from rates × players, forced
  monotone, and projected by blending a weighted log-time fit with a
  proportional-to-players path. On a flat history, earners grow like
  the pool raised to a rate-dependent power below one (common trophies
  nearly keep pace with new players; rare and near-saturated ones lag,
  which is what dilution looks like). Projections are capped by a
  tier-dependent saturation share of the future pool — platinum
  saturates lowest, bronze highest.

The median rate is projected earners over projected players. When at
least three epochs exist, it is blended in logit space with a direct
per-trophy logit-trajectory fit against log-time.

### Honest spread on the logit scale

Quantiles are `sigmoid(logit(median) + z · sd)`. The spread grows with
the square root of the horizon in days but keeps a high floor, because
even a 90-minute horizon can straddle a refresh that moves every rate
at once. It widens further when epochs are scarce, for rare tiers
(platinum, gold), for rates near 0% or 100%, for trophies with tiny
earner counts, and when the observed trajectory is wobbly.

### Edge cases

Empty or missing history falls back to tier priors with a wide spread;
single reads, rates of exactly 0 or 100, null rates, missing player
counts and malformed history entries are all handled without crashing,
and the program always emits a valid output file.

## Usage

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each output file is `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`
with one row per trophy in the case, quantiles non-decreasing and
within 0–100.

## License

MIT — see `LICENSE`.
