"""Pocket-pass classifier — logistic regression vs GBM comparison.

Training strategy (semi-supervised with weak labels)
-----------------------------------------------------
Gold labels cover 7 Argentina matches.  Spatial candidates in those matches
that were *not* explicitly flagged True are treated as *weak negatives* (the
detector over-generated them; the user saw nothing in the video).

    Positives  (y=1): gold True  ∩ spatial candidates   [high-confidence]
    Negatives  (y=0): spatial candidates in gold matches - gold True - ambiguous
    Excluded:  ambiguous labels

Cross-validation: GroupKFold with k = n_gold_matches (leave-one-match-out).

Outputs
-------
outputs/pocket_clf_report.txt     -- CV comparison table + feature importances
outputs/pocket_passes_scored.csv  -- all 2226 candidates with lr_prob, gbm_prob
outputs/pocket_active_query.csv   -- 30 most uncertain cases (GBM) to label next
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

OUT_DIR = ROOT / "outputs"

NUMERIC_FEATS = [
    "depth_behind",
    "iso_dist",
    "end_x",
    "abs_end_y",
    "length",
    "line_x",
    "n_line",
    "abs_start_y",
    "run_mag",
    "start_x",
    "passer_vs_line",
    "passer_past_line",  # binary: passer already past defensive line
]
CAT_FEATS = ["pocket_type", "ballHeight"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_end_y"]        = df["end_y"].abs()
    df["abs_start_y"]      = df["start_y"].abs()
    df["passer_vs_line"]   = df["start_x"] - df["line_x"]
    df["passer_past_line"] = (df["passer_vs_line"] > 0).astype(float)
    dummies = pd.get_dummies(df[CAT_FEATS], drop_first=False, dtype=float)
    return pd.concat([df[NUMERIC_FEATS], dummies], axis=1)


def load_training(pp_path: Path, gold_path: Path):
    pp   = pd.read_csv(pp_path)
    gold = pd.read_csv(gold_path)
    pp["match_id"]   = pp["match_id"].astype(str)
    gold["match_id"] = gold["match_id"].astype(str)

    cands = pp[pp["match_id"].isin(set(gold["match_id"]))].copy()

    gold_tf = gold[gold["label"].isin(["True", "False"])]
    cands = cands.merge(gold_tf[["match_id", "match_min", "label"]],
                        on=["match_id", "match_min"], how="left")
    cands["y"] = (cands["label"] == "True").astype(int)

    excl = set(zip(
        gold[gold["label"] == "ambiguous"]["match_id"],
        gold[gold["label"] == "ambiguous"]["match_min"],
    ))
    cands = cands[
        ~cands.apply(lambda r: (r["match_id"], r["match_min"]) in excl, axis=1)
    ].copy()

    X      = build_features(cands)
    y      = cands["y"]
    groups = cands["match_id"]
    return X, y, groups


def make_lr() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced",
                                   max_iter=1000, C=1.0, random_state=42)),
    ])


def make_gbm() -> HistGradientBoostingClassifier:
    # HistGBM handles NaN natively; no imputer/scaler needed.
    # class_weight="balanced" supported since sklearn 1.2.
    return HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )


def run_cv(model, X, y, groups, k, scoring):
    return cross_validate(model, X, y, groups=groups,
                          cv=GroupKFold(n_splits=k),
                          scoring=scoring, return_train_score=False)


def print_cv(name: str, res: dict, k: int):
    print(f"\n{name}  (k={k} leave-one-match-out CV):")
    for label, key in [("AUC", "test_AUC"), ("Precision", "test_precision"),
                       ("Recall",  "test_recall"), ("F1", "test_f1")]:
        v = res[key]
        print(f"  {label:10s}: {v.mean():.3f}  +/- {v.std():.3f}"
              f"  [{' '.join(f'{x:.2f}' for x in v)}]")


def main():
    pp_path   = OUT_DIR / "pocket_passes.csv"
    gold_path = OUT_DIR / "pocket_gold_labels.csv"

    print("=" * 64)
    print("Pocket-pass classifier: LR vs GBM")
    print("=" * 64)

    X, y, groups = load_training(pp_path, gold_path)
    pos = int(y.sum()); neg = len(y) - pos
    k   = groups.nunique()
    print(f"\nTraining: {len(y)} rows  positives={pos}  negatives={neg}"
          f"  ({k} gold matches, all Argentina)")

    scoring = {
        "AUC":       "roc_auc",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall":    make_scorer(recall_score,    zero_division=0),
        "f1":        make_scorer(f1_score,        zero_division=0),
    }

    lr_res  = run_cv(make_lr(),  X, y, groups, k, scoring)
    gbm_res = run_cv(make_gbm(), X, y, groups, k, scoring)

    print_cv("Logistic Regression", lr_res, k)
    print_cv("HistGBM            ", gbm_res, k)

    # ── Train final models on all labelled data ───────────────────────────────
    lr_pipe = make_lr();  lr_pipe.fit(X, y)
    gbm_clf = make_gbm(); gbm_clf.fit(X, y)

    # LR coefficients
    lr      = lr_pipe.named_steps["clf"]
    coef_df = (pd.DataFrame({"feature": X.columns, "coef": lr.coef_[0]})
               .assign(abs_coef=lambda d: d["coef"].abs())
               .sort_values("abs_coef", ascending=False))
    print("\nLR top coefficients:")
    print(coef_df.head(12)[["feature", "coef"]].to_string(index=False))

    # GBM feature importances via permutation (model-agnostic, more reliable)
    X_imp = X.fillna(X.median())   # HistGBM handles NaN but permutation_importance needs clean array
    perm  = permutation_importance(gbm_clf, X_imp, y, n_repeats=20,
                                   random_state=42, scoring="roc_auc")
    imp_df = (pd.DataFrame({"feature":    X.columns,
                             "importance": perm.importances_mean,
                             "std":        perm.importances_std})
              .sort_values("importance", ascending=False))
    print("\nGBM feature importances (permutation, AUC drop):")
    print(imp_df.head(12)[["feature", "importance", "std"]].to_string(index=False))

    # ── Score all 2226 candidates ─────────────────────────────────────────────
    pp_all = pd.read_csv(pp_path)
    pp_all["match_id"] = pp_all["match_id"].astype(str)
    X_all  = build_features(pp_all).reindex(columns=X.columns, fill_value=0.0)

    pp_all["lr_prob"]  = lr_pipe.predict_proba(X_all)[:, 1]
    pp_all["gbm_prob"] = gbm_clf.predict_proba(X_all)[:, 1]
    # ensemble: simple average
    pp_all["ml_probability"] = (pp_all["lr_prob"] + pp_all["gbm_prob"]) / 2

    scored_path = OUT_DIR / "pocket_passes_scored.csv"
    pp_all.to_csv(scored_path, index=False)
    print(f"\nScored {len(pp_all)} candidates -> {scored_path.name}")

    for col, name in [("lr_prob", "LR"), ("gbm_prob", "GBM"),
                      ("ml_probability", "Ensemble")]:
        counts = {t: int((pp_all[col] >= t).sum()) for t in [0.3, 0.5, 0.7, 0.9]}
        print(f"  {name:8s} thresholds: {counts}")

    # ── Active-learning query (GBM uncertainty) ───────────────────────────────
    gold      = pd.read_csv(gold_path)
    gold_mids = set(gold["match_id"].astype(str))
    unlabelled = pp_all[~pp_all["match_id"].isin(gold_mids)].copy()
    unlabelled["uncertainty"] = (unlabelled["gbm_prob"] - 0.5).abs()
    query = (unlabelled.sort_values("uncertainty")
             .head(30)[["match_id", "match_min", "team", "opponent",
                         "passer", "receiver", "pocket_type",
                         "gbm_prob", "depth_behind", "iso_dist"]])
    query.to_csv(OUT_DIR / "pocket_active_query.csv", index=False)
    print("\nActive-learning query (30 most uncertain, GBM) -> pocket_active_query.csv")
    print(query[["match_id", "match_min", "pocket_type", "gbm_prob"]
                ].head(10).to_string(index=False))

    # ── Write report ──────────────────────────────────────────────────────────
    rpath = OUT_DIR / "pocket_clf_report.txt"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write("Pocket-pass classifier: LR vs GBM\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training: {pos} positives  {neg} negatives (weak)  "
                f"{k} gold matches (all Argentina)\n\n")
        for name, res in [("Logistic Regression", lr_res), ("HistGBM", gbm_res)]:
            f.write(f"{name}:\n")
            for label, key in [("AUC", "test_AUC"), ("Precision", "test_precision"),
                                ("Recall", "test_recall"), ("F1", "test_f1")]:
                v = res[key]
                f.write(f"  {label:10s}: {v.mean():.3f} +/- {v.std():.3f}\n")
            f.write("\n")
        f.write("LR coefficients:\n")
        f.write(coef_df.head(12)[["feature", "coef"]].to_string(index=False))
        f.write("\n\nGBM feature importances (permutation, AUC drop):\n")
        f.write(imp_df.head(12)[["feature", "importance", "std"]].to_string(index=False))
        f.write(f"\n\nEnsemble (avg) scored {len(pp_all)} candidates"
                f" -> pocket_passes_scored.csv\n")
    print(f"\nReport -> {rpath.name}")


if __name__ == "__main__":
    main()
