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

# config: paths and column definitions
DATA_DIR = Path("data")
STAGE2_DIR = DATA_DIR / "stage2_artifacts"
OUTPUT_DIR = DATA_DIR / "stage3_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# load feature names from stage 2
with open(STAGE2_DIR / "econml_feature_names.json", "r") as f:
    FEATURE_NAMES = json.load(f)

def load_econml_data(stage2_dir: Path) -> dict:
    data = {}
    for key in ["X", "W"]:
        data[key] = pd.read_parquet(stage2_dir / f"econml_{key}.parquet")
    for key in ["T", "Y"]:
        data[key] = np.load(stage2_dir / f"econml_{key}.npy").astype(np.float64)
    return data

def train_causal_forest(X: pd.DataFrame, T: np.ndarray, Y: np.ndarray,
                        seed: int = 1) -> CausalForestDML:

    # train causal forest dml with lightgbm nuisance models
    model_y = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=seed, verbose=-1)
    model_t = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed, verbose=-1)
    
    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=10,
        max_depth=8,
        random_state=seed,
        verbose=0,
    )
    
    # fit without W
    cf.fit(Y, T, X=X.reset_index(drop=True))  # removed W=W argument
    return cf

def estimate_heterogeneous_effects(cf: CausalForestDML, X: pd.DataFrame,
                                   n_samples: int = 10000, seed: int = 1) -> pd.DataFrame:
    np.random.seed(seed)
    sample_idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sample = X.iloc[sample_idx]

    tau = cf.effect(X_sample)
    ci_low, ci_high = cf.effect_interval(X_sample, alpha=0.05)

    results = pd.DataFrame({
        "cate_point": tau,
        "cate_ci_low": ci_low,
        "cate_ci_high": ci_high,
        "significant": (ci_low > 0) | (ci_high < 0)
    }, index=X_sample.index)

    return results

def main():
    logger.info("Loading Stage 2 EconML datasets...")
    data = load_econml_data(STAGE2_DIR)
    X, T, Y = data["X"], data["T"], data["Y"]
    
    # restrict to at-risk borrowers
    at_risk_mask = (Y > 0) | (T > 0)
    X_risk = X[at_risk_mask].reset_index(drop=True)
    T_risk = T[at_risk_mask]
    Y_risk = Y[at_risk_mask]
    
    logger.info(f"At-risk sample: {at_risk_mask.sum():,} loans ({at_risk_mask.mean():.2%} of total)")
    logger.info(f"Treatment rate in at-risk sample: {T_risk.mean():.2%}")
    logger.info(f"Outcome rate in at-risk sample: {Y_risk.mean():.2%}")
    
    logger.info("Training Causal Forest DML...")
    cf = train_causal_forest(X_risk, T_risk, Y_risk)  # removed W_risk
    
    logger.info("Estimating heterogeneous treatment effects...")
    cate_results = estimate_heterogeneous_effects(cf, X_risk)
    
    # save artifacts
    joblib.dump(cf, OUTPUT_DIR / "causal_forest_model.pkl")
    cate_results.to_parquet(OUTPUT_DIR / "cate_estimates.parquet")
    
    # summary stats
    logger.info(f"CATE mean: {cate_results['cate_point'].mean():.4f}")
    logger.info(f"CATE std: {cate_results['cate_point'].std():.4f}")
    logger.info(f"Significant effects: {cate_results['significant'].mean():.2%}")
    logger.info(f"Beneficial effects (CATE < 0): {(cate_results['cate_point'] < 0).mean():.2%}")
    logger.info("Stage 3a complete. Artifacts saved to %s", OUTPUT_DIR)

if __name__ == "__main__":
    main()