import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
STAGE2_DIR = DATA_DIR / "stage2_artifacts"
STAGE3_DIR = DATA_DIR / "stage3_artifacts"

# load stage 1 parquet (contains loan_sequence_number)
df = pd.read_parquet(DATA_DIR / "freddie_mac_causal_ready.parquet")

# apply same at-risk filter used in stage 2
at_risk = (df["ever_seriously_delinquent"] == 1) | (df["ever_treated"] == 1)
df_atrisk = df[at_risk].reset_index(drop=True)

# if trim mask exists from stage 3a, apply it to align with causal forest training
trim_mask_path = STAGE3_DIR / "trim_mask.npy"
if trim_mask_path.exists():
    trim_mask = np.load(trim_mask_path)
    # trim_mask was computed on at-risk sample, so apply directly
    df_trimmed = df_atrisk[trim_mask].reset_index(drop=True)
    print(f"Applied propensity trimming: {len(df_trimmed):,} loans retained")
else:
    df_trimmed = df_atrisk
    print("No trim mask found; using at-risk sample only")

# extract loan_sequence_number and save as DataFrame (double brackets!)
loan_ids = df_trimmed[["loan_sequence_number"]].reset_index(drop=True)
loan_ids.to_parquet(STAGE2_DIR / "loan_ids.parquet")

print(f"Saved {len(loan_ids)} loan IDs to {STAGE2_DIR / 'loan_ids.parquet'}")