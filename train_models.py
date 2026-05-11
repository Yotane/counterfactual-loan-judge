import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import shap
import duckdb
import joblib
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, r2_score, mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "freddie_mac_causal_ready.parquet"
OUTPUT_DIR = DATA_DIR / "stage2_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = ["occupancy_status", "loan_purpose", "property_state"]
# ever_defaulted: binary foreclosure proxy (zero balance codes 02, 03, 09)
TARGET_COL = "ever_defaulted"
TREATMENT_COL = "ever_treated"

PROPENSITY_FEATURES = [
    "credit_score", "original_ltv", "original_dti", "original_upb",
    "original_interest_rate", "num_units", "original_loan_term",
    "high_ltv", "high_dti", "low_credit_score", "risk_score",
    "occupancy_status", "loan_purpose", "property_state"
]

def load_and_encode(df: pd.DataFrame) -> tuple[pd.DataFrame, OneHotEncoder]:
    # one-hot encode categorical features for tree models and econml
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_cats = encoder.fit_transform(df[CATEGORICAL_COLS])
    cat_cols_encoded = encoder.get_feature_names_out(CATEGORICAL_COLS)

    df_cats = pd.DataFrame(encoded_cats, columns=cat_cols_encoded, index=df.index)
    df_numeric = df.drop(columns=CATEGORICAL_COLS)

    df_encoded = pd.concat([df_numeric, df_cats], axis=1)
    return df_encoded, encoder

def build_modeling_features(encoder: OneHotEncoder, propensity_features: list, categorical_cols: list) -> list:
    # numeric props + ohe columns
    ohe_cols = list(encoder.get_feature_names_out(categorical_cols))
    numeric_cols = [c for c in propensity_features if c not in categorical_cols]
    return numeric_cols + ohe_cols

def train_propensity_model(df: pd.DataFrame, feature_cols: list, seed: int = 1) -> lgb.LGBMClassifier:
    # train lightgbm to estimate p(t=1|x) for causal adjustment
    X = df[feature_cols]
    y = df[TREATMENT_COL]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        is_unbalance=True,
        random_state=seed,
        verbose=-1
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    logger.info(f"Propensity model AUC-ROC: {auc:.4f}")

    return model

def outcome_objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> float:
    # optuna search space for outcome model — classifier for binary ever_defaulted
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "is_unbalance": True,
        "verbose": -1,
        "random_state": 1
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred)

def train_outcome_model(df: pd.DataFrame, feature_cols: list, n_trials: int = 50,
                        seed: int = 1) -> tuple[lgb.LGBMClassifier, dict]:
    # optimize lightgbm classifier with optuna, train final model, compute metrics
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    logger.info("Running Optuna hyperparameter search...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: outcome_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False
    )

    logger.info(f"Best Optuna AUC: {study.best_value:.4f}")

    best_params = study.best_params
    best_params.update({"is_unbalance": True, "verbose": -1, "random_state": seed})

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict_proba(X_val)[:, 1]

    metrics = {
        "auc_roc": float(roc_auc_score(y_val, y_pred)),
        "avg_precision": float(average_precision_score(y_val, y_pred)),
        "optuna_best_params": best_params
    }
    logger.info(f"Outcome model metrics: {metrics}")
    return final_model, metrics

def generate_shap_explanations(model: lgb.LGBMClassifier, df: pd.DataFrame,
                                feature_cols: list, n_samples: int = 5000) -> pd.DataFrame:
    # shap values for model interpretability and stakeholder reporting
    sample_df = df[feature_cols].sample(n_samples, random_state=1)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_df)

    # shap_values is a list for classifiers, take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    shap_df["base_value"] = explainer.expected_value if not isinstance(
        explainer.expected_value, list) else explainer.expected_value[1]
    return shap_df

def prepare_econml_datasets(df: pd.DataFrame, feature_cols: list) -> dict:
    # restrict to at-risk loans: ever seriously delinquent or ever treated
    at_risk = (df["ever_seriously_delinquent"] == 1) | (df[TREATMENT_COL] == 1)
    df_atrisk = df[at_risk].copy()
    logger.info(f"at-risk sample: {len(df_atrisk):,} loans, "
                f"treatment rate: {df_atrisk[TREATMENT_COL].mean():.2%}")
    logger.info(f"foreclosure rate in at-risk: {df_atrisk[TARGET_COL].mean():.2%}")

    # refit propensity model on at-risk population so scores reflect this subpopulation
    prop_atrisk = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, is_unbalance=True, random_state=1, verbose=-1
    )
    X_prop = df_atrisk[feature_cols]
    T_prop = df_atrisk[TREATMENT_COL]
    prop_atrisk.fit(X_prop, T_prop)
    df_atrisk = df_atrisk.copy()
    df_atrisk["propensity_score"] = prop_atrisk.predict_proba(X_prop)[:, 1]
    logger.info(f"at-risk propensity score mean: {df_atrisk['propensity_score'].mean():.4f}")

    # x: heterogeneity drivers — risk features cate is expected to vary across
    # w: full controls including propensity score for dml residualization
    x_cols = [
        "credit_score", "original_ltv", "original_dti", "original_upb",
        "high_ltv", "high_dti", "low_credit_score", "risk_score"
    ]
    X = df_atrisk[x_cols]
    T = df_atrisk[TREATMENT_COL].values
    Y = df_atrisk[TARGET_COL].values
    W = df_atrisk[feature_cols].copy()
    W["propensity_score"] = df_atrisk["propensity_score"].values

    logger.info(f"X shape: {X.shape}, W shape: {W.shape}")
    logger.info(f"EconML ready: T mean={T.mean():.4f}, Y mean={Y.mean():.4f}")
    return {"X": X, "T": T, "Y": Y, "W": W, "feature_names": x_cols}

def sql_bi_export(df: pd.DataFrame, output_path: Path):
    # duckdb query for power bi/tableau export: foreclosure risk segmentation
    con = duckdb.connect()

    query = """
    SELECT
        property_state,
        occupancy_status,
        loan_purpose,
        COUNT(*) as loan_count,
        AVG(credit_score) as avg_credit_score,
        AVG(original_ltv) as avg_ltv,
        AVG(original_dti) as avg_dti,
        SUM(ever_treated) as treated_count,
        SUM(ever_defaulted) as foreclosure_count,
        ROUND(100.0 * SUM(ever_treated) / COUNT(*), 2) as treatment_rate_pct,
        ROUND(100.0 * SUM(ever_defaulted) / COUNT(*), 2) as foreclosure_rate_pct
    FROM df
    GROUP BY property_state, occupancy_status, loan_purpose
    HAVING loan_count >= 50
    ORDER BY foreclosure_rate_pct DESC
    """

    result = con.execute(query).df()
    result.to_csv(output_path / "bi_segmentation_export.csv", index=False)
    logger.info(f"BI export saved to {output_path / 'bi_segmentation_export.csv'}")

def main():
    logger.info("Loading stage 1 parquet...")
    df = pd.read_parquet(INPUT_PATH)
    df = df.dropna(subset=[TARGET_COL, TREATMENT_COL] + PROPENSITY_FEATURES)

    logger.info(f"Target distribution — ever_defaulted mean: {df[TARGET_COL].mean():.4f}")

    df_for_bi = df.copy()

    logger.info("Encoding categorical features...")
    df_encoded, encoder = load_and_encode(df)

    modeling_features = build_modeling_features(encoder, PROPENSITY_FEATURES, CATEGORICAL_COLS)
    logger.info(f"Modeling features count: {len(modeling_features)}")

    logger.info("Training propensity model...")
    prop_model = train_propensity_model(df_encoded, modeling_features)
    df_encoded["propensity_score"] = prop_model.predict_proba(df_encoded[modeling_features])[:, 1]

    logger.info("Training outcome model with Optuna...")
    outcome_model, metrics = train_outcome_model(df_encoded, modeling_features, n_trials=50)

    logger.info("Generating SHAP explanations...")
    shap_df = generate_shap_explanations(outcome_model, df_encoded, modeling_features)

    logger.info("Preparing EconML datasets...")
    econml_data = prepare_econml_datasets(df_encoded, modeling_features)

    logger.info("Running DuckDB BI export...")
    sql_bi_export(df_for_bi, OUTPUT_DIR)

    # save artifacts
    joblib.dump(prop_model, OUTPUT_DIR / "propensity_model.pkl")
    joblib.dump(outcome_model, OUTPUT_DIR / "outcome_model.pkl")
    joblib.dump(encoder, OUTPUT_DIR / "categorical_encoder.pkl")

    with open(OUTPUT_DIR / "outcome_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    shap_df.to_parquet(OUTPUT_DIR / "shap_values.parquet")
    for key, val in econml_data.items():
        if isinstance(val, pd.DataFrame):
            val.to_parquet(OUTPUT_DIR / f"econml_{key}.parquet")
        elif isinstance(val, np.ndarray):
            np.save(OUTPUT_DIR / f"econml_{key}.npy", val)
        else:
            with open(OUTPUT_DIR / f"econml_{key}.json", "w") as f:
                json.dump(val, f)

    logger.info("Stage 2 complete. Artifacts saved to %s", OUTPUT_DIR)

if __name__ == "__main__":
    main()