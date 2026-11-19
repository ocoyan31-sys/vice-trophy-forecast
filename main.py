"""Vice Trophy Forecaster.

Forecasts GTA VI (PS5) trophy earn rates on Exophase at a future target
instant, as five quantiles per trophy, scored by pinball loss on the
logit scale.

Model outline
-------------
Exophase does not update continuously: the tracked-player count and
the rates stay frozen across many house reads and then jump at a
refresh.  The history is therefore collapsed into distinct refresh
epochs before anything is fitted.

Inside an epoch a rate is earners / players_tracked, and those two
parts move very differently at a refresh: the denominator can multiply
in one step (diluting almost every rate) while earners only ever grow.
So the forecaster projects the two parts separately:

* players_tracked is projected with a weighted fit against log-time,
  which bends the early launch surge into slower later growth;
* each trophy's earners are projected with their own log-time fit,
  blended with a proportional-to-players path, forced monotone, and
  capped by a tier-dependent saturation share of the future pool
  (platinum saturates lowest, bronze highest).

The median rate is projected earners over projected players.  When at
least three epochs exist, that median is blended in logit space with a
direct per-trophy logit-trajectory fit.  Uncertainty is a spread on
the logit scale that grows with the square root of the horizon in
days and widens further for scarce epochs, rare tiers, near-boundary
rates and tiny earner counts: an unseen refresh can move a rate a
long way, and the pinball loss rewards an honest spread.

Pure Python 3 standard library, no network, no randomness: two runs
on the same case produce byte-identical output.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

LEVEL_Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
           0.6744897501960817, 1.6448536269514722)

# Prior median earn rate (percent) when a case carries no usable
# history for a trophy, keyed by tier.
TIER_PRIOR_RATE = {
    "platinum": 0.45,
    "gold": 1.8,
    "silver": 4.5,
    "bronze": 11.0,
}

# Long-run share of tracked players who can plausibly hold a trophy
# of each tier inside the forecast window.  Projected earners are
# never allowed past this share of the projected player pool.
TIER_CAP_FRACTION = {
    "platinum": 0.16,
    "gold": 0.48,
    "silver": 0.68,
    "bronze": 0.94,
}

# Extra logit spread per tier: rarer tiers are harder to pin down.
TIER_SPREAD_BONUS = {
    "platinum": 0.30,
    "gold": 0.16,
    "silver": 0.07,
    "bronze": 0.0,
}


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def logit_of_percent(rate_pct):
    """Logit of a percentage rate, clamped to the scoring range."""
    p = clamp(rate_pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def percent_of_logit(logit_value):
    """Inverse of logit_of_percent, returned as a percentage."""
    if logit_value >= 0.0:
        z = math.exp(-logit_value)
        p = 1.0 / (1.0 + z)
    else:
        z = math.exp(logit_value)
        p = z / (1.0 + z)
    return p * 100.0


def parse_time(text):
    """Parse an ISO-8601 timestamp to epoch seconds, else None."""
    if not isinstance(text, str) or not text:
        return None
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def weighted_line_fit(xs, ys, ws):
    """Weighted least squares fit y ~= a + b*x, or None if degenerate."""
    total_w = sum(ws)
    if total_w <= 0.0 or len(xs) < 2:
        return None
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total_w
    mean_y = sum(w * y for y, w in zip(ys, ws)) / total_w
    var_x = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, ws))
    if var_x <= 1e-12:
        return (mean_y, 0.0)
    cov = sum(w * (x - mean_x) * (y - mean_y)
              for x, y, w in zip(xs, ys, ws))
    slope = cov / var_x
    return (mean_y - slope * mean_x, slope)


def recency_weights(count):
    """Mildly increasing weights: later epochs describe the game better."""
    if count <= 1:
        return [1.0] * max(count, 0)
    return [0.55 + 0.9 * (i / (count - 1.0)) for i in range(count)]


def collapse_epochs(history):
    """Collapse consecutive identical reads into refresh epochs.

    Returns a time-sorted list of dicts with the read time in epoch
    seconds, the players_tracked count (positive float or None) and
    the raw rates list exactly as printed.  Reads without a usable
    timestamp keep their order on a synthetic half-hour ladder.
    """
    epochs = []
    last_key = None
    for index, read in enumerate(history):
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list):
            rates = []
        players = read.get("players_tracked")
        if isinstance(players, (int, float)) and players > 0:
            players_value = float(players)
        else:
            players_value = None
        earned_total = read.get("trophies_earned")
        key = (players_value,
               earned_total if isinstance(earned_total, (int, float)) else None,
               tuple(rates))
        moment = parse_time(read.get("at"))
        if moment is None:
            moment = parse_time(read.get("slot"))
        if moment is None:
            moment = 1800.0 * index
        if key == last_key and epochs:
            continue
        last_key = key
        epochs.append({"time": moment, "players": players_value,
                       "rates": rates})
    epochs.sort(key=lambda item: item["time"])
    return epochs


def project_players(epochs, target_time, horizon_days):
    """Project players_tracked at the target instant.

    With at least two distinct player counts, a weighted fit of
    players against ln(1 + hours since the first epoch) is
    extrapolated, clamped to stay at or above the last count and to
    avoid absurd explosions.  With one flat epoch, a generic launch
    growth factor rising with the square root of the horizon applies.
    Returns (projected players, growth factor over the last count),
    or (None, 1.0) when no epoch carries a player count.
    """
    known = [(ep["time"], ep["players"]) for ep in epochs
             if ep["players"] is not None]
    if not known:
        return None, 1.0
    last_players = known[-1][1]
    distinct = {value for _, value in known}
    if len(known) >= 2 and len(distinct) >= 2:
        origin = known[0][0]
        xs = [math.log1p(max(0.0, (t - origin) / 3600.0)) for t, _ in known]
        ys = [value for _, value in known]
        fit = weighted_line_fit(xs, ys, recency_weights(len(known)))
        if fit is not None:
            intercept, slope = fit
            if slope > 0.0:
                x_target = math.log1p(max(0.0, (target_time - origin) / 3600.0))
                predicted = intercept + slope * x_target
                upper = last_players * 60.0 + 5000.0
                predicted = clamp(predicted, last_players, upper)
                return predicted, predicted / last_players
    safe_days = max(horizon_days, 0.0)
    growth = 1.0 + 0.85 * math.sqrt(safe_days) + 0.15 * safe_days
    return last_players * growth, growth


def earner_series(epochs, index):
    """Monotone earner counts for one trophy across the epochs.

    Earners are rebuilt as rate/100 * players_tracked and replaced by
    their running maximum, because a player who earned a trophy keeps
    it even when a later printed rate wobbles down on rounding.
    Returns a list of (time, earners, players) triples.
    """
    series = []
    running_max = 0.0
    for ep in epochs:
        players = ep["players"]
        if players is None or index >= len(ep["rates"]):
            continue
        rate = ep["rates"][index]
        if not isinstance(rate, (int, float)):
            continue
        earners = clamp(float(rate), 0.0, 100.0) / 100.0 * players
        running_max = max(running_max, earners)
        series.append((ep["time"], running_max, players))
    return series


def project_earners(series, target_time, growth, players_target,
                    cap_fraction):
    """Project one trophy's earner count at the target instant.

    With a time span of epochs, a weighted log-time fit of earners is
    blended with a proportional path (earners scaling with the player
    pool, i.e. a constant rate).  With a single flat epoch, earners
    grow like the pool raised to a rate-dependent power below one:
    common trophies nearly keep pace with new players, rare ones lag,
    which is exactly the dilution seen at a refresh.  The projection
    never decreases and never exceeds the tier cap share of the pool.
    """
    last_time, last_earners, last_players = series[-1]
    cap_earners = cap_fraction * players_target
    predicted = None
    if len(series) >= 2 and series[-1][0] > series[0][0]:
        origin = series[0][0]
        xs = [math.log1p(max(0.0, (t - origin) / 3600.0)) for t, _, _ in series]
        ys = [value for _, value, _ in series]
        fit = weighted_line_fit(xs, ys, recency_weights(len(series)))
        if fit is not None:
            intercept, slope = fit
            x_target = math.log1p(max(0.0, (target_time - origin) / 3600.0))
            if slope <= 0.0:
                linear_pred = last_earners
            else:
                linear_pred = intercept + slope * x_target
            linear_pred = clamp(linear_pred, last_earners,
                                last_earners * 40.0 + 50.0)
            proportional = last_earners * growth
            predicted = 0.55 * linear_pred + 0.45 * proportional
    if predicted is None:
        last_rate = 100.0 * last_earners / last_players if last_players > 0 else 0.0
        if last_rate >= 90.0:
            # A near-total rate on a flat history is a small-sample
            # artefact of the first tracked cohort: the incoming pool
            # cannot all hold the trophy, so earners lag far behind
            # pool growth and the rate dilutes hard at a refresh.
            alpha = 0.50
        elif last_rate >= 50.0:
            alpha = 0.70
        elif last_rate >= 20.0:
            alpha = 0.85
        elif last_rate >= 5.0:
            alpha = 0.85
        elif last_rate >= 1.0:
            alpha = 0.78
        elif last_rate > 0.0:
            alpha = 0.70
        else:
            alpha = 0.62
        if last_earners <= 0.0:
            predicted = 0.6 * (growth ** alpha)
        else:
            predicted = last_earners * (growth ** alpha)
    floor = last_earners if last_earners > 0.0 else 0.0
    return clamp(predicted, floor, cap_earners)


def last_rate_for(epochs, index):
    """Most recent printed rate for a trophy, searching backwards."""
    for ep in reversed(epochs):
        if index < len(ep["rates"]):
            rate = ep["rates"][index]
            if isinstance(rate, (int, float)):
                return float(rate)
    return None


def logit_trajectory(epochs, index, target_time):
    """Fit logit(rate) against log-time; return (prediction, resid_sd).

    Uses every epoch with a printed rate for this trophy.  Returns
    (None, 0.0) when there are too few points or no time span.
    """
    points = []
    for ep in epochs:
        if index < len(ep["rates"]):
            rate = ep["rates"][index]
            if isinstance(rate, (int, float)):
                points.append((ep["time"], logit_of_percent(float(rate))))
    if len(points) < 3 or points[-1][0] <= points[0][0]:
        return None, 0.0
    origin = points[0][0]
    xs = [math.log1p(max(0.0, (t - origin) / 3600.0)) for t, _ in points]
    ys = [value for _, value in points]
    ws = recency_weights(len(points))
    fit = weighted_line_fit(xs, ys, ws)
    if fit is None:
        return None, 0.0
    intercept, slope = fit
    x_target = math.log1p(max(0.0, (target_time - origin) / 3600.0))
    prediction = intercept + slope * x_target
    sq = 0.0
    wsum = 0.0
    for x, y, w in zip(xs, ys, ws):
        resid = y - (intercept + slope * x)
        sq += w * resid * resid
        wsum += w
    resid_sd = math.sqrt(sq / wsum) if wsum > 0.0 else 0.0
    return prediction, resid_sd


def spread_for(horizon_days, epoch_count, tier, median_rate,
               last_earners, resid_sd):
    """Honest logit-scale spread for one trophy's forecast.

    The base width grows with sqrt(horizon days) but keeps a high
    floor: even a 90-minute practice horizon straddles a refresh that
    can move every rate at once.  Scarce epochs, rare tiers, rates
    near 0 or 100, tiny earner counts and a wobbly observed trajectory
    all widen it further.  The total is capped so the quantiles still
    carry information.
    """
    safe_days = max(horizon_days, 0.0)
    sd = 0.82 + 0.34 * math.sqrt(safe_days)
    if epoch_count <= 1:
        sd += 0.45
    elif epoch_count == 2:
        sd += 0.30
    elif epoch_count == 3:
        sd += 0.16
    sd += TIER_SPREAD_BONUS.get(tier, 0.0)
    if median_rate < 1.0:
        sd += 0.25
    elif median_rate < 3.0:
        sd += 0.12
    if median_rate > 90.0:
        sd += 0.35
    elif median_rate > 75.0:
        sd += 0.15
    if last_earners is not None:
        if last_earners < 5.0:
            sd += 0.35
        elif last_earners < 20.0:
            sd += 0.20
        elif last_earners < 60.0:
            sd += 0.08
    if resid_sd > 0.0:
        sd += clamp(0.55 * resid_sd, 0.0, 0.55)
    return clamp(sd, 0.85, 2.60)


def forecast_case(case):
    """Build the output object {"trophies": [...]} for one case."""
    trophies = case.get("trophies")
    if not isinstance(trophies, list):
        trophies = []
    history = case.get("history")
    if not isinstance(history, list):
        history = []
    epochs = collapse_epochs(history)

    checkpoint = parse_time(case.get("at"))
    if checkpoint is None and epochs:
        checkpoint = epochs[-1]["time"]
    if checkpoint is None:
        checkpoint = 0.0
    target_time = parse_time(case.get("target"))
    horizon_minutes = case.get("horizon_minutes")
    if target_time is None:
        if isinstance(horizon_minutes, (int, float)):
            target_time = checkpoint + float(horizon_minutes) * 60.0
        else:
            target_time = checkpoint + 90.0 * 60.0
    if isinstance(horizon_minutes, (int, float)) and horizon_minutes > 0:
        horizon_days = float(horizon_minutes) / 1440.0
    else:
        horizon_days = max(0.0, (target_time - checkpoint) / 86400.0)

    players_target, growth = project_players(epochs, target_time,
                                             horizon_days)
    known_players = [ep["players"] for ep in epochs if ep["players"] is not None]
    players_last = known_players[-1] if known_players else None
    if players_target is None:
        # No player counts anywhere: work against a nominal pool so
        # the tier priors still produce sensible percentages.
        players_target = 2000.0
        players_last = 2000.0
        growth = 1.0

    rows = []
    for index, trophy in enumerate(trophies):
        if isinstance(trophy, dict):
            raw_id = trophy.get("id")
            tier = str(trophy.get("tier") or "bronze").lower()
        else:
            raw_id = None
            tier = "bronze"
        if tier not in TIER_PRIOR_RATE:
            tier = "bronze"
        try:
            trophy_id = int(raw_id)
        except (TypeError, ValueError):
            trophy_id = index + 1

        series = earner_series(epochs, index)
        printed = last_rate_for(epochs, index)
        resid_sd = 0.0
        if series:
            cap_fraction = TIER_CAP_FRACTION[tier]
            if printed is not None:
                # A trophy already above its tier cap keeps headroom:
                # the cap is a saturation guide, not a hard ceiling
                # on something the page has already printed.
                cap_fraction = max(cap_fraction,
                                   clamp(printed, 0.0, 100.0) / 100.0 * 1.15)
                cap_fraction = min(cap_fraction, 0.995)
            earners_target = project_earners(series, target_time, growth,
                                             players_target, cap_fraction)
            median_rate = 100.0 * earners_target / players_target
            last_earners = series[-1][1]
            if len(epochs) >= 3:
                traj_logit, resid_sd = logit_trajectory(epochs, index,
                                                        target_time)
                if traj_logit is not None:
                    blended = 0.65 * logit_of_percent(median_rate) \
                        + 0.35 * traj_logit
                    median_rate = percent_of_logit(blended)
        elif printed is not None:
            # Rates exist but no player counts to rebuild earners:
            # hold the last printed rate as the median.
            median_rate = clamp(printed, 0.01, 99.99)
            last_earners = None
        else:
            median_rate = TIER_PRIOR_RATE[tier]
            last_earners = None

        median_rate = clamp(median_rate, 0.02, 99.2)
        sd = spread_for(horizon_days, len(epochs), tier, median_rate,
                        last_earners, resid_sd)
        centre = logit_of_percent(median_rate)
        quantiles = []
        running = 0.0
        for z in LEVEL_Z:
            value = percent_of_logit(centre + z * sd)
            value = clamp(round(value, 4), 0.0, 100.0)
            running = max(running, value)
            quantiles.append(running)
        rows.append({"id": trophy_id, "q": quantiles})
    return {"trophies": rows}


def fallback_output(case):
    """Tier-prior output used if anything unexpected goes wrong."""
    trophies = case.get("trophies") if isinstance(case, dict) else None
    if not isinstance(trophies, list):
        trophies = []
    rows = []
    for index, trophy in enumerate(trophies):
        tier = "bronze"
        trophy_id = index + 1
        if isinstance(trophy, dict):
            tier = str(trophy.get("tier") or "bronze").lower()
            try:
                trophy_id = int(trophy.get("id"))
            except (TypeError, ValueError):
                pass
        centre = logit_of_percent(TIER_PRIOR_RATE.get(tier, 11.0))
        quantiles = []
        running = 0.0
        for z in LEVEL_Z:
            value = clamp(round(percent_of_logit(centre + z * 1.7), 4),
                          0.0, 100.0)
            running = max(running, value)
            quantiles.append(running)
        rows.append({"id": trophy_id, "q": quantiles})
    return {"trophies": rows}


def main(argv=None):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError:
        pass
    paths = sorted(glob.glob(os.path.join(input_dir, "*.in")))
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as handle:
                case = json.load(handle)
            if not isinstance(case, dict):
                case = {}
        except (OSError, ValueError):
            case = {}
        try:
            result = forecast_case(case)
        except Exception:
            result = fallback_output(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        out_path = os.path.join(output_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
