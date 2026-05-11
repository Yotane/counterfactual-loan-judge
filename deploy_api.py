from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# config: paths
DATA_DIR = Path("data")
STAGE2_DIR = DATA_DIR / "stage2_artifacts"
STAGE3_DIR = DATA_DIR / "stage3_artifacts"

# business parameters — same as policy_simulation.py
COST_INTERVENTION = 300
DOLLAR_COST_PER_FORECLOSURE = 60000
TARGETING_PERCENTILE = 0.20

# load models at startup
logger.info("Loading models...")
causal_forest = joblib.load(STAGE3_DIR / "causal_forest_model.pkl")
encoder = joblib.load(STAGE2_DIR / "categorical_encoder.pkl")

with open(STAGE2_DIR / "econml_feature_names.json", "r") as f:
    FEATURE_NAMES = json.load(f)

with open(STAGE2_DIR / "outcome_metrics.json", "r") as f:
    OUTCOME_METRICS = json.load(f)

# load loan ID mapping for GET endpoints (loan_index → loan_sequence_number)
try:
    LOAN_IDS = pd.read_parquet(STAGE2_DIR / "loan_ids.parquet")["loan_sequence_number"].values
    logger.info(f"Loaded {len(LOAN_IDS)} loan IDs for API responses")
except FileNotFoundError:
    LOAN_IDS = None
    logger.warning("loan_ids.parquet not found; loan_id will be null in GET responses")

# x_cols used during causal forest training (8 risk features, not full 71)
X_COLS = [
    "credit_score", "original_ltv", "original_dti", "original_upb",
    "high_ltv", "high_dti", "low_credit_score", "risk_score"
]

app = FastAPI(
    title="Loan Modification Targeting API",
    description=(
        "Causal ML system for mortgage servicers. "
        "Ranks distressed borrowers by how much they benefit from loan modification, "
        "enabling budget-constrained outreach that maximizes foreclosures prevented."
    ),
    version="1.0"
)

#  request/response schemas 

class LoanFeatures(BaseModel):
    loan_id: Optional[str] = Field(None, description="Servicer loan reference ID")
    credit_score: int = Field(..., ge=300, le=850)
    original_ltv: float = Field(..., ge=0, le=200)
    original_dti: float = Field(..., ge=0, le=65)
    original_upb: float = Field(..., gt=0)
    original_interest_rate: float = Field(..., gt=0)
    num_units: int = Field(..., ge=1, le=4)
    original_loan_term: int = Field(..., gt=0)
    occupancy_status: str = Field(..., description="P=Primary, I=Investment, S=Second Home")
    loan_purpose: str = Field(..., description="P=Purchase, C=Cash-out Refi, N=No Cash-out Refi")
    property_state: str = Field(..., description="Two-letter state code e.g. CA, TX")

class ModificationRecommendation(BaseModel):
    loan_id: Optional[str]
    recommend_modification: bool
    cate: float = Field(..., description="Estimated causal effect on foreclosure probability")
    foreclosure_probability_reduction: float = Field(
        ..., description="Estimated percentage point reduction in foreclosure probability"
    )
    expected_savings_dollars: float = Field(
        ..., description="Expected dollar value of prevented foreclosure loss"
    )
    expected_net_benefit_dollars: float = Field(
        ..., description="Expected savings minus intervention cost"
    )
    priority_tier: str = Field(
        ..., description="HIGH / MEDIUM / LOW — based on CATE percentile"
    )
    rationale: str

class BatchLoanRequest(BaseModel):
    loans: list[LoanFeatures]

class BatchRecommendation(BaseModel):
    total_loans: int
    recommended_count: int
    total_expected_savings_dollars: float
    loans: list[ModificationRecommendation]

#  preprocessing 

def engineer_features(loan: LoanFeatures) -> dict:
    # compute derived risk features — must match ETL pipeline exactly
    high_ltv = 1 if loan.original_ltv > 80 else 0
    high_dti = 1 if loan.original_dti > 43 else 0
    low_credit_score = 1 if loan.credit_score < 620 else 0
    investment_property = 1 if loan.occupancy_status == "I" else 0
    cash_out_refi = 1 if loan.loan_purpose == "C" else 0
    risk_score = (
        high_ltv * 0.25 + high_dti * 0.25 +
        low_credit_score * 0.3 + investment_property * 0.1 +
        cash_out_refi * 0.1
    )
    return {
        "credit_score": loan.credit_score,
        "original_ltv": loan.original_ltv,
        "original_dti": loan.original_dti,
        "original_upb": loan.original_upb,
        "original_interest_rate": loan.original_interest_rate,
        "num_units": loan.num_units,
        "original_loan_term": loan.original_loan_term,
        "high_ltv": high_ltv,
        "high_dti": high_dti,
        "low_credit_score": low_credit_score,
        "risk_score": risk_score,
        "occupancy_status": loan.occupancy_status,
        "loan_purpose": loan.loan_purpose,
        "property_state": loan.property_state
    }

def preprocess_loan(loan: LoanFeatures) -> pd.DataFrame:
    # engineer features, encode categoricals, return X_cols only for causal forest
    features = engineer_features(loan)
    df = pd.DataFrame([features])

    cat_cols = ["occupancy_status", "loan_purpose", "property_state"]
    encoded_cats = encoder.transform(df[cat_cols])
    cat_df = pd.DataFrame(
        encoded_cats,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index
    )
    numeric_df = df.drop(columns=cat_cols)
    df_full = pd.concat([numeric_df, cat_df], axis=1)

    # causal forest was trained on X_COLS only (8 risk features)
    return df_full[X_COLS]

def score_to_recommendation(loan: LoanFeatures, cate: float) -> ModificationRecommendation:
    # cate is negative when modification helps (reduces foreclosure probability)
    foreclosure_reduction_pct = abs(cate) * 100 if cate < 0 else 0.0
    expected_savings = abs(cate) * DOLLAR_COST_PER_FORECLOSURE if cate < 0 else 0.0
    net_benefit = expected_savings - COST_INTERVENTION

    # priority tier based on CATE magnitude
    # thresholds derived from at-risk population CATE distribution
    # top 10%: cate < -0.05, top 20%: cate < -0.03, rest: cate >= -0.03
    if cate < -0.05:
        priority_tier = "HIGH"
    elif cate < -0.03:
        priority_tier = "MEDIUM"
    else:
        priority_tier = "LOW"

    recommend = net_benefit > 0

    if recommend:
        rationale = (
            f"Modification estimated to reduce foreclosure probability by "
            f"{foreclosure_reduction_pct:.1f} pct points. "
            f"Expected net benefit: ${net_benefit:,.0f} after ${COST_INTERVENTION} outreach cost."
        )
    else:
        rationale = (
            f"Estimated effect ({foreclosure_reduction_pct:.1f} pct point reduction) "
            f"does not justify ${COST_INTERVENTION} outreach cost at current foreclosure cost assumptions."
        )

    return ModificationRecommendation(
        loan_id=loan.loan_id,
        recommend_modification=recommend,
        cate=float(cate),
        foreclosure_probability_reduction=round(foreclosure_reduction_pct, 3),
        expected_savings_dollars=round(expected_savings, 2),
        expected_net_benefit_dollars=round(net_benefit, 2),
        priority_tier=priority_tier,
        rationale=rationale
    )

#  endpoints 

@app.post("/score_loan", response_model=ModificationRecommendation,
          summary="Score a single loan for modification priority")
async def score_loan(loan: LoanFeatures):
    """
    Submit a single distressed loan. Returns:
    - Whether to recommend modification outreach
    - Estimated causal effect on foreclosure probability
    - Expected dollar savings vs intervention cost
    - Priority tier (HIGH / MEDIUM / LOW)
    """
    try:
        X = preprocess_loan(loan)
        cate = float(causal_forest.effect(X)[0])
        return score_to_recommendation(loan, cate)
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score_batch", response_model=BatchRecommendation,
          summary="Score a batch of loans and rank by modification priority")
async def score_batch(request: BatchLoanRequest):
    """
    Submit a portfolio of distressed loans. Returns:
    - All loans scored and ranked by expected benefit
    - Total expected savings across recommended loans
    - Sorted by CATE (most responsive first) for budget allocation
    """
    if len(request.loans) == 0:
        raise HTTPException(status_code=400, detail="No loans provided")
    if len(request.loans) > 10000:
        raise HTTPException(status_code=400, detail="Batch limit is 10,000 loans")

    try:
        # preprocess all loans
        X_list = [preprocess_loan(loan) for loan in request.loans]
        X_batch = pd.concat(X_list, ignore_index=True)

        # score all at once
        cates = causal_forest.effect(X_batch)

        # build recommendations
        recommendations = [
            score_to_recommendation(loan, float(cate))
            for loan, cate in zip(request.loans, cates)
        ]

        # sort by CATE ascending (most negative = highest priority first)
        recommendations.sort(key=lambda r: r.cate)

        recommended = [r for r in recommendations if r.recommend_modification]
        total_savings = sum(r.expected_savings_dollars for r in recommended)

        return BatchRecommendation(
            total_loans=len(recommendations),
            recommended_count=len(recommended),
            total_expected_savings_dollars=round(total_savings, 2),
            loans=recommendations
        )
    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top_loans",
         summary="Get top N loans most likely to benefit from modification")
async def get_top_loans(
    n: int = 100,
    min_priority: str = "LOW"
):
    """
    Returns the top N distressed loans ranked by expected benefit from modification.
    Use this to decide who to call first with your outreach team.
    
    Parameters:
    - n: number of loans to return (default 100, max 1000)
    - min_priority: filter by tier — HIGH, MEDIUM, or LOW (default LOW = all)
    """
    if n > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 loans per request")

    try:
        # load the ranked policy output from policy_simulation.py
        policy_df = pd.read_parquet(STAGE3_DIR / "optimal_policy.parquet")
        X = pd.read_parquet(STAGE2_DIR / "econml_X.parquet")

        # load trim mask to align indices
        mask = np.load(STAGE3_DIR / "trim_mask.npy")
        X_risk = X[mask].reset_index(drop=True)

        # sort by ranking score descending (most negative CATE first)
        ranked = policy_df.sort_values("ranking_score", ascending=False).copy()
        ranked = ranked.join(X_risk[["credit_score", "original_ltv",
                                      "original_dti", "original_upb"]])

        # filter by priority tier
        tier_map = {"HIGH": -0.05, "MEDIUM": -0.03, "LOW": 1.0}  # 1.0 includes all CATEs
        cate_threshold = tier_map.get(min_priority.upper(), 1.0)
        ranked = ranked[ranked["cate_prob"] <= cate_threshold]  # keep loans with CATE <= threshold

        top_n = ranked.head(n)

        results = []
        for idx, row in top_n.iterrows():
            loan_id = str(LOAN_IDS[int(idx)]) if LOAN_IDS is not None and int(idx) < len(LOAN_IDS) else None
            results.append({
                "rank": len(results) + 1,
                "loan_id": loan_id,  # ← ADDED: Freddie Mac loan sequence number
                "loan_index": int(idx),
                "cate": round(float(row["cate_prob"]), 4),
                "foreclosure_probability_reduction_pct": round(
                    float(row["foreclosures_prevented"]) * 100, 2
                ),
                "expected_savings_dollars": round(
                    float(row["foreclosures_prevented"]) * DOLLAR_COST_PER_FORECLOSURE, 0
                ),
                "expected_net_benefit_dollars": round(
                    float(row["expected_utility_dollars"]), 0
                ),
                "priority_tier": (
                    "HIGH" if row["cate_prob"] < -0.05
                    else "MEDIUM" if row["cate_prob"] < -0.03
                    else "LOW"
                ),
                "credit_score": int(row["credit_score"]),
                "original_ltv": round(float(row["original_ltv"]), 1),
                "original_dti": round(float(row["original_dti"]), 1),
                "original_upb": round(float(row["original_upb"]), 0)
            })

        return {
            "total_returned": len(results),
            "min_priority_filter": min_priority.upper(),
            "total_expected_savings_dollars": round(
                sum(r["expected_savings_dollars"] for r in results), 0
            ),
            "total_net_benefit_dollars": round(
                sum(r["expected_net_benefit_dollars"] for r in results), 0
            ),
            "loans": results
        }

    except Exception as e:
        logger.error(f"Top loans error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/budget_plan",
         summary="How many foreclosures can we prevent with a given outreach budget?")
async def budget_plan(budget_dollars: float = 100000):
    """
    Given a dollar budget for modification outreach, returns:
    - How many loans to target
    - Which ones (ranked by benefit)
    - Expected foreclosures prevented
    - Expected ROI
    
    Example: budget_dollars=500000 means you can contact 1,666 borrowers at $300 each.
    """
    if budget_dollars <= 0:
        raise HTTPException(status_code=400, detail="Budget must be positive")

    max_loans = int(budget_dollars // COST_INTERVENTION)
    if max_loans == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Budget too small. Minimum is ${COST_INTERVENTION} per loan."
        )

    try:
        policy_df = pd.read_parquet(STAGE3_DIR / "optimal_policy.parquet")
        X = pd.read_parquet(STAGE2_DIR / "econml_X.parquet")
        mask = np.load(STAGE3_DIR / "trim_mask.npy")
        X_risk = X[mask].reset_index(drop=True)

        # rank by most beneficial first
        ranked = policy_df.sort_values("ranking_score", ascending=False).copy()
        ranked = ranked.join(X_risk[["credit_score", "original_ltv",
                                      "original_dti", "original_upb"]])
        within_budget = ranked.head(max_loans)

        total_foreclosures_prevented = float(
            within_budget["foreclosures_prevented"].sum()
        )
        total_savings = total_foreclosures_prevented * DOLLAR_COST_PER_FORECLOSURE
        total_cost = len(within_budget) * COST_INTERVENTION
        roi = total_savings - total_cost

        # tier breakdown
        high = (within_budget["cate_prob"] < -0.05).sum()
        medium = ((within_budget["cate_prob"] >= -0.05) &
                  (within_budget["cate_prob"] < -0.03)).sum()
        low = (within_budget["cate_prob"] >= -0.03).sum()

        top_5 = []
        for idx, row in within_budget.head(5).iterrows():
            loan_id = str(LOAN_IDS[int(idx)]) if LOAN_IDS is not None and int(idx) < len(LOAN_IDS) else None
            top_5.append({
                "rank": len(top_5) + 1,
                "loan_id": loan_id,  # ← ADDED: Freddie Mac loan sequence number
                "cate": round(float(row["cate_prob"]), 4),
                "foreclosure_reduction_pct": round(
                    float(row["foreclosures_prevented"]) * 100, 2
                ),
                "expected_net_benefit_dollars": round(
                    float(row["expected_utility_dollars"]), 0
                ),
                "credit_score": int(row["credit_score"]),
                "original_upb": round(float(row["original_upb"]), 0)
            })

        return {
            "budget_dollars": budget_dollars,
            "loans_targeted": len(within_budget),
            "outreach_cost_dollars": total_cost,
            "expected_foreclosures_prevented": round(total_foreclosures_prevented, 1),
            "expected_savings_dollars": round(total_savings, 0),
            "expected_roi_dollars": round(roi, 0),
            "roi_multiple": round(total_savings / total_cost, 1) if total_cost > 0 else 0,
            "priority_breakdown": {
                "HIGH": int(high),
                "MEDIUM": int(medium),
                "LOW": int(low)
            },
            "top_5_loans": top_5,
            "note": (
                f"Loans ranked by CATE (causal effect size). "
                f"Top {TARGETING_PERCENTILE:.0%} of at-risk portfolio recommended. "
                f"Assumptions: ${COST_INTERVENTION} per outreach, "
                f"${DOLLAR_COST_PER_FORECLOSURE:,} average foreclosure cost."
            )
        }

    except Exception as e:
        logger.error(f"Budget plan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio_summary",
         summary="Summary statistics for the at-risk portfolio")
async def portfolio_summary():
    """
    Returns model performance metrics and business parameter assumptions.
    Use this to understand the model's calibration before making outreach decisions.
    """
    return {
        "model_performance": OUTCOME_METRICS,
        "business_parameters": {
            "cost_per_intervention_dollars": COST_INTERVENTION,
            "assumed_foreclosure_cost_dollars": DOLLAR_COST_PER_FORECLOSURE,
            "targeting_percentile": TARGETING_PERCENTILE
        },
        "priority_tier_thresholds": {
            "HIGH": "CATE < -0.05 (top ~10% most responsive)",
            "MEDIUM": "CATE < -0.03 (top ~20% most responsive)",
            "LOW": "CATE >= -0.03 (below targeting threshold)"
        },
        "model_training": {
            "dataset": "Freddie Mac SFLLD 2015 vintage",
            "loans_in_training": "80,109 at-risk borrowers",
            "causal_question": "Does modification prevent foreclosure among distressed borrowers?",
            "avg_cate": -0.0226,
            "top_10pct_cate": -0.0917
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)