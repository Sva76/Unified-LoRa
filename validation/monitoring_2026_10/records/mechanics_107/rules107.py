"""Prospective, one-class exploratory rules. Standard library; no model API."""
import math
import statistics as S

RECOVERED = "RECOVERED_WITHIN_50"
NOT_RECOVERED = "NO_CONFIRMED_RECOVERY_WITHIN_50"
MODELS = ("stable_examples", "mean_loss", "mean_loss_variability", "low_loss_examples", "always_recover")


def average_ranks(values):
    return [1 + sum(v < x for v in values) + (sum(v == x for v in values)-1)/2 for x in values]


def features_at(t, states, rule):
    # Explicitly ignore every observation after the common cutoff.
    window = [r for r in states if t-19 <= r["state"] <= t]
    if [r["state"] for r in window] != list(range(t-19, t+1)):
        raise ValueError("Incomplete common 20-state feature window")
    for r in window:
        if not r["valid"] or len(r["eval_per_example"]) != 32:
            raise ValueError("Invalid feature observation")
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v >= 0
                   for v in r["eval_per_example"]+[r["eval_loss"]]):
            raise ValueError("Invalid feature loss")
    def rise(v):
        return max(0.0, S.median(v[-5:])-S.median(v[:15]))
    per_example = [rise([r["eval_per_example"][i] for r in window]) for i in range(32)]
    mean = [r["eval_loss"] for r in window]
    return {
        "stable_examples": S.median(per_example[i] for i in rule["stable_indices_0based"]),
        "mean_loss": rise(mean),
        "mean_loss_variability": max(0.0, S.pstdev(mean[-5:])-S.pstdev(mean[:15])),
        "low_loss_examples": S.median(per_example[i] for i in rule["low_loss_indices_0based"]),
    }


def decision_at(t, states, regular, past_decisions, rule):
    row = {"state":t, "feature_cutoff_state":t, "episode":None}
    if t > rule["last_episode_start"]:
        return dict(row, reason="FOLLOWUP_ONLY")
    previous = [d["state"] for d in past_decisions if d["state"] < t and d.get("episode")]
    if previous and t <= previous[-1]+rule["horizon_updates"]:
        return dict(row, reason="EPISODE_WINDOW_ACTIVE")
    scores = {g["state"]:g["primary_score"] for g in regular if g["state"] <= t}
    if any(s not in scores for s in (t-30,t-20,t-10,t)):
        return dict(row, reason="REFERENCE_NOT_AVAILABLE")
    reference = S.median(scores[s] for s in (t-30,t-20,t-10))
    if reference < rule["minimum_reference_correct"]:
        return dict(row, reason="INSUFFICIENT_HEADROOM")
    drop = reference-scores[t]
    if drop < rule["drop_answers"]:
        return dict(row, reason="NO_QUALIFYING_DROP")
    features = features_at(t,states,rule)
    predictions = {m:NOT_RECOVERED if value > rule["thresholds"][m] else RECOVERED
                   for m,value in features.items()}
    predictions["always_recover"] = RECOVERED
    return dict(row,reason="PREDICTION_COMMITTED",episode={
        "start_state":t,"reference":reference,"observed_score":scores[t],"observed_drop":drop,
        "drop_answers":rule["drop_answers"],"horizon_end":t+rule["horizon_updates"],
        "recovery_target":reference-rule["recovery_tolerance_answers"],
        "feature_window":[t-19,t],"features":features,"predictions":predictions})


def outcome(episode,regular,completed_updates,rule):
    scores = {g["state"]:g["primary_score"] for g in regular}
    t,end,target = episode["start_state"],episode["horizon_end"],episode["recovery_target"]
    expected = list(range(t+10,end+1,10))
    confirmed = [s for s in expected[1:] if s in scores and s-10 in scores
                 and scores[s] >= target and scores[s-10] >= target]
    full = completed_updates >= end and all(s in scores for s in expected)
    label = (RECOVERED if confirmed else NOT_RECOVERED) if full else "CENSORED"
    return {"outcome":label,"confirmation_state":confirmed[0] if confirmed else None,
            "future_scores":[{"state":s,"score":scores.get(s)} for s in expected]}


def counts(episodes):
    result = {}
    for m in MODELS:
        recovered = [e for e in episodes if e["outcome"] == RECOVERED]
        not_recovered = [e for e in episodes if e["outcome"] == NOT_RECOVERED]
        fp = sum(e["predictions"][m] == NOT_RECOVERED for e in recovered)
        tp = sum(e["predictions"][m] == NOT_RECOVERED for e in not_recovered)
        result[m] = {"observed_recovered":len(recovered),"observed_not_recovered":len(not_recovered),
                     "false_nonrecovery_predictions":fp,"correct_recovery_predictions":len(recovered)-fp,
                     "correct_nonrecovery_predictions":tp,"missed_nonrecoveries":len(not_recovered)-tp,
                     "censored_excluded":sum(e["outcome"] == "CENSORED" for e in episodes)}
    return result
