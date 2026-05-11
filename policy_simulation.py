import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
STAGE2_DIR = DATA_DIR / "stage2_artifacts"
STAGE3_DIR = DATA_DIR / "stage3_artifacts"
OUTPUT_DIR = STAGE3_DIR

# servicer outreach cost per loan and average foreclosure cost to servicer
COST_INTERVENTION = 300
DOLLAR_COST_PER_FORECLOSURE = 60000

# realistic servicer capacity constraint:
# servicers cannot modify all distressed loans simultaneously
# target top 20% most responsive borrowers by CATE magnitude
TARGETING_PERCENTILE = 0.20

def load_artifacts() -> dict:
    # load causal forest model and cate estimates
    cf = joblib.load(STAGE3_DIR / "causal_forest_model.pkl")
    cate = pd.read_parquet(STAGE3_DIR / "cate_estimates.parquet")

    X = pd.read_parquet(STAGE2_DIR / "econml_X.parquet")
    T = np.load(STAGE2_DIR / "econml_T.npy").astype(np.float64)
    Y = np.load(STAGE2_DIR / "econml_Y.npy").astype(np.float64)

    # apply same trimming mask used during training
    mask = np.load(STAGE3_DIR / "trim_mask.npy")
    X = X[mask].reset_index(drop=True)
    T = T[mask]
    Y = Y[mask]

    logger.info(f"At-risk sample loaded: {len(X):,} loans")
    logger.info(f"Foreclosure rate: {Y.mean():.2%}")
    return {"cf": cf, "cate": cate, "X": X, "T": T, "Y": Y}

def compute_utility(cate: np.ndarray) -> np.ndarray:
    # expected savings = probability of foreclosure prevented × cost per foreclosure
    # cate is negative when modification helps — take abs for savings calculation
    foreclosures_prevented = np.where(cate < 0, np.abs(cate), 0.0)
    return foreclosures_prevented * DOLLAR_COST_PER_FORECLOSURE - COST_INTERVENTION

def optimize_treatment_assignment(X: pd.DataFrame, cf) -> pd.DataFrame:
    cate = cf.effect(X)
    utility = compute_utility(cate)

    # rank by CATE — most negative = highest priority = most responsive to modification
    # target top TARGETING_PERCENTILE by CATE magnitude, not just utility > 0
    # this reflects real servicer capacity constraints
    threshold = np.percentile(cate, TARGETING_PERCENTILE * 100)
    optimal_treatment = (cate <= threshold).astype(int)

    # ranking score for sorting: higher = better candidate for modification
    ranking_score = -cate

    return pd.DataFrame({
        "optimal_treatment": optimal_treatment,
        "expected_utility_dollars": utility,
        "cate_prob": cate,
        "foreclosures_prevented": np.where(cate < 0, np.abs(cate), 0.0),
        "ranking_score": ranking_score,
        "ranking_decile": pd.qcut(ranking_score, q=10, labels=False, duplicates="drop")
    }, index=X.index)

def simulate_counterfactual_outcomes(Y: np.ndarray, T: np.ndarray,
                                     optimal_T: np.ndarray,
                                     cate: np.ndarray) -> dict:
    # y_cf = what would Y have been under optimal policy
    y_cf = Y + cate * (optimal_T - T)
    y_cf = np.clip(y_cf, 0, 1)

    return {
        "observed_foreclosure_rate": float(Y.mean()),
        "counterfactual_foreclosure_rate": float(y_cf.mean()),
        "estimated_reduction_pct_points": float((Y.mean() - y_cf.mean()) * 100),
        "treatment_rate_observed": float(T.mean()),
        "treatment_rate_optimal": float(optimal_T.mean()),
        "total_foreclosures_prevented": float(max(Y.sum() - y_cf.sum(), 0)),
        "estimated_total_savings_dollars": float(
            max(Y.sum() - y_cf.sum(), 0) * DOLLAR_COST_PER_FORECLOSURE
        )
    }

def generate_decile_report(policy_df: pd.DataFrame,
                            X: pd.DataFrame, feature_names: list) -> list:
    # decile 9 = top 10% most likely to benefit — highest priority for outreach
    # decile 0 = bottom 10% — least responsive, lowest priority
    df = X.copy()
    df["cate_prob"] = policy_df["cate_prob"].values
    df["expected_utility"] = policy_df["expected_utility_dollars"].values
    df["foreclosures_prevented"] = policy_df["foreclosures_prevented"].values
    df["ranking_decile"] = policy_df["ranking_decile"].values
    df["recommended"] = policy_df["optimal_treatment"].values

    top5_numeric = [f for f in feature_names[:5]
                    if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

    rules = []
    for decile in sorted(df["ranking_decile"].dropna().unique(), reverse=True):
        segment = df[df["ranking_decile"] == decile]
        rule = {
            "decile": int(decile),
            "avg_cate_prob": float(segment["cate_prob"].mean()),
            "avg_foreclosures_prevented": float(segment["foreclosures_prevented"].mean()),
            "avg_expected_utility_dollars": float(segment["expected_utility"].mean()),
            "segment_size": int(len(segment)),
            "pct_recommended": float(segment["recommended"].mean()),
            "recommend_intervention": bool(segment["expected_utility"].mean() > 0),
            "avg_features": {
                k: float(v) for k, v in segment[top5_numeric].mean().items()
            }
        }
        rules.append(rule)

    return rules

def budget_constrained_targeting(policy_df: pd.DataFrame,
                                  budget_dollars: float) -> dict:
    # rank by expected benefit, target top-n within budget
    # mirrors line yahoo use case: fixed message quota, maximize re-engagement
    ranked = policy_df.sort_values("ranking_score", ascending=False).copy()
    ranked["cum_cost"] = COST_INTERVENTION * (np.arange(len(ranked)) + 1)
    within_budget = ranked[ranked["cum_cost"] <= budget_dollars]

    return {
        "budget_dollars": float(budget_dollars),
        "loans_targeted": int(len(within_budget)),
        "foreclosures_prevented": float(within_budget["foreclosures_prevented"].sum()),
        "total_cost": float(len(within_budget) * COST_INTERVENTION),
        "estimated_roi_dollars": float(
            within_budget["foreclosures_prevented"].sum() * DOLLAR_COST_PER_FORECLOSURE
            - len(within_budget) * COST_INTERVENTION
        )
    }

def main():
    logger.info("Loading Stage 3 artifacts...")
    artifacts = load_artifacts()
    cf, cate_df, X, T, Y = (artifacts["cf"], artifacts["cate"],
                              artifacts["X"], artifacts["T"], artifacts["Y"])

    logger.info(f"Cost per intervention: ${COST_INTERVENTION}")
    logger.info(f"Dollar cost per foreclosure: ${DOLLAR_COST_PER_FORECLOSURE:,}")
    logger.info(f"Targeting top {TARGETING_PERCENTILE:.0%} most responsive borrowers")

    logger.info("Computing optimal treatment assignment...")
    policy_df = optimize_treatment_assignment(X, cf)

    recommended = policy_df[policy_df["optimal_treatment"] == 1]
    not_recommended = policy_df[policy_df["optimal_treatment"] == 0]

    logger.info(f"Loans recommended for intervention: {len(recommended):,} "
                f"({len(recommended)/len(policy_df):.2%})")
    logger.info(f"Avg CATE for recommended loans: {recommended['cate_prob'].mean():.4f}")
    logger.info(f"Avg CATE for non-recommended loans: {not_recommended['cate_prob'].mean():.4f}")
    logger.info(f"Avg foreclosures prevented (recommended): "
                f"{recommended['foreclosures_prevented'].mean():.4f}")
    logger.info(f"Avg expected utility (recommended): "
                f"${recommended['expected_utility_dollars'].mean():,.0f}")

    logger.info("Simulating counterfactual outcomes...")
    simulation = simulate_counterfactual_outcomes(
        Y, T,
        policy_df["optimal_treatment"].values,
        policy_df["cate_prob"].values
    )

    logger.info("Generating decile report...")
    with open(STAGE2_DIR / "econml_feature_names.json", "r") as f:
        feature_names = json.load(f)
    rules = generate_decile_report(policy_df, X, feature_names)

    # budget-constrained scenarios
    budget_results = {}
    for budget in [100_000, 500_000, 1_000_000]:
        budget_results[f"budget_{budget}"] = budget_constrained_targeting(
            policy_df, float(budget)
        )
        logger.info(f"Budget ${budget:,}: "
                    f"{budget_results[f'budget_{budget}']['loans_targeted']:,} loans targeted, "
                    f"ROI ${budget_results[f'budget_{budget}']['estimated_roi_dollars']:,.0f}")

    # save outputs
    policy_df.to_parquet(OUTPUT_DIR / "optimal_policy.parquet")

    with open(OUTPUT_DIR / "policy_simulation.json", "w") as f:
        json.dump(simulation, f, indent=2)

    with open(OUTPUT_DIR / "policy_rules.json", "w") as f:
        json.dump(rules, f, indent=2)

    with open(OUTPUT_DIR / "budget_scenarios.json", "w") as f:
        json.dump(budget_results, f, indent=2)

    # summary for README

    logger.info(f"Observed foreclosure rate: {simulation['observed_foreclosure_rate']:.2%}")
    logger.info(f"Optimal treatment rate: {simulation['treatment_rate_optimal']:.2%}")
    logger.info(f"Avg CATE for targeted loans: {recommended['cate_prob'].mean():.4f}")
    logger.info(f"Avg CATE for non-targeted loans: {not_recommended['cate_prob'].mean():.4f}")
    logger.info(f"CATE ratio top/bottom: "
                f"{recommended['cate_prob'].mean() / not_recommended['cate_prob'].mean():.1f}x")
    logger.info("--- Budget ROI Scenarios ---")
    for budget in [100_000, 500_000, 1_000_000]:
        b = budget_results[f"budget_{budget}"]
        logger.info(f"  ${budget:,} budget → {b['loans_targeted']:,} loans → "
                    f"ROI ${b['estimated_roi_dollars']:,.0f}")

if __name__ == "__main__":
    main()