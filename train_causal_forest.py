import pandas as pd
import numpy as np
import lightgbm as lgb
from econml.dml import CausalForestDML
import joblib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
STAGE2_DIR = DATA_DIR / "stage2_artifacts"
OUTPUT_DIR = DATA_DIR / "stage3_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(STAGE2_DIR / "econml_feature_names.json", "r") as f:
    FEATURE_NAMES = json.load(f)

# lower bound 0.02 not 0.05: treatment rate ~16% in at-risk so strict lower discards too many controls
# upper bound 0.90: severely distressed borrowers often have high mod probability
PS_LOWER = 0.02
PS_UPPER = 0.90

def load_econml_data(stage2_dir: Path) -> dict:
    # load x, t, y, w from stage 2 parquet/npy files
    data = {}
    data["X"] = pd.read_parquet(stage2_dir / "econml_X.parquet")
    data["W"] = pd.read_parquet(stage2_dir / "econml_W.parquet")
    for key in ["T", "Y"]:
        data[key] = np.load(stage2_dir / f"econml_{key}.npy").astype(np.float64)
    return data

def apply_propensity_trimming(X, T, Y, W, ps_lower=PS_LOWER, ps_upper=PS_UPPER):
    # trim to overlap region using propensity scores saved in w from stage 2
    # asymmetric bounds handle treatment rate in at-risk mortgage subpopulation
    if "propensity_score" in W.columns:
        ps = W["propensity_score"].values
        mask = (ps > ps_lower) & (ps < ps_upper)
        logger.info(f"Propensity trimming [{ps_lower}, {ps_upper}]: {mask.mean():.2%} of loans retained ({mask.sum():,})")
        logger.info(f"Treatment rate in trimmed sample: {T[mask].mean():.2%}")
    else:
        mask = (Y > 0) | (T > 0)
        logger.warning("No propensity scores found; using at-risk filter")

    return (
        X[mask].reset_index(drop=True),
        T[mask],
        Y[mask],
        W[mask].reset_index(drop=True),
        mask
    )

def train_causal_forest(X: pd.DataFrame, T: np.ndarray, Y: np.ndarray,
                        W: pd.DataFrame, seed: int = 1) -> CausalForestDML:
    # model_y: regressor even for binary outcome — econml residualizes y as continuous
    # model_t: predicts treatment probability from all controls
    model_y = lgb.LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=seed, verbose=-1
    )
    model_t = lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        is_unbalance=True, random_state=seed, verbose=-1
    )

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=20,
        max_depth=6,
        random_state=seed,
        verbose=0,
    )

    # pass w so dml residualizes on controls before building the forest
    cf.fit(Y, T, X=X.reset_index(drop=True), W=W.reset_index(drop=True))
    return cf

def estimate_heterogeneous_effects(cf: CausalForestDML, X: pd.DataFrame,
                                   n_samples: int = 10000, seed: int = 1) -> pd.DataFrame:
    # negative cate = modification reduces foreclosure probability (good)
    np.random.seed(seed)
    sample_idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]

    tau = cf.effect(X_sample)
    ci_low, ci_high = cf.effect_interval(X_sample, alpha=0.05)

    results = pd.DataFrame({
        "cate_point": tau,
        "cate_ci_low": ci_low,
        "cate_ci_high": ci_high,
        # significant if ci does not cross zero
        "significant": (ci_low > 0) | (ci_high < 0),
        "beneficial": tau < 0
    }, index=X_sample.index)

    return results

def main():
    logger.info("Loading Stage 2 EconML datasets...")
    data = load_econml_data(STAGE2_DIR)
    X, T, Y, W = data["X"], data["T"], data["Y"], data["W"]

    logger.info(f"Y (ever_defaulted) — mean: {Y.mean():.4f}")
    logger.info(f"T (ever_treated) — mean: {T.mean():.4f}")

    X_trim, T_trim, Y_trim, W_trim, mask = apply_propensity_trimming(X, T, Y, W)

    logger.info(f"Trimmed sample: {len(X_trim):,} loans")
    logger.info(f"Treatment rate in trimmed sample: {T_trim.mean():.2%}")
    logger.info(f"Foreclosure rate in trimmed sample: {Y_trim.mean():.2%}")

    logger.info("Training Causal Forest DML...")
    cf = train_causal_forest(X_trim, T_trim, Y_trim, W_trim)

    logger.info("Estimating heterogeneous treatment effects...")
    cate_results = estimate_heterogeneous_effects(cf, X_trim)

    # save artifacts
    joblib.dump(cf, OUTPUT_DIR / "causal_forest_model.pkl")
    cate_results.to_parquet(OUTPUT_DIR / "cate_estimates.parquet")
    np.save(OUTPUT_DIR / "trim_mask.npy", mask)

    # negative cate = modification reduces foreclosure probability (good)
    logger.info(f"CATE mean: {cate_results['cate_point'].mean():.4f}")
    logger.info(f"CATE std: {cate_results['cate_point'].std():.4f}")
    logger.info(f"Significant effects: {cate_results['significant'].mean():.2%}")
    logger.info(f"Beneficial effects (CATE < 0): {cate_results['beneficial'].mean():.2%}")
    logger.info(f"Top 10% most responsive — avg CATE: "
                f"{cate_results['cate_point'].nsmallest(int(len(cate_results)*0.1)).mean():.4f}")
    logger.info("Stage 3a complete. Artifacts saved to %s", OUTPUT_DIR)

if __name__ == "__main__":
    main()