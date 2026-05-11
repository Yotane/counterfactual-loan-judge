import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, when, avg, count, max as spark_max, min as spark_min,
    first, last, lag, lead, sum as spark_sum
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType
)

def main():
    spark = SparkSession.builder \
        .appName("freddie_mac_sfll_etl") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.host", "localhost") \
        .config("spark.driver.bindAddress", "localhost") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    try:
        orig_schema = StructType([
            StructField("credit_score", IntegerType(), True),
            StructField("first_payment_date", StringType(), True),
            StructField("first_time_homebuyer", StringType(), True),
            StructField("maturity_date", StringType(), True),
            StructField("msa", StringType(), True),
            StructField("mi_percentage", IntegerType(), True),
            StructField("num_units", IntegerType(), True),
            StructField("occupancy_status", StringType(), True),
            StructField("original_cltv", DoubleType(), True),
            StructField("original_dti", DoubleType(), True),
            StructField("original_upb", DoubleType(), True),
            StructField("original_ltv", DoubleType(), True),
            StructField("original_interest_rate", DoubleType(), True),
            StructField("channel", StringType(), True),
            StructField("ppm_flag", StringType(), True),
            StructField("amortization_type", StringType(), True),
            StructField("property_state", StringType(), True),
            StructField("property_type", StringType(), True),
            StructField("postal_code", StringType(), True),
            StructField("loan_sequence_number", StringType(), True),
            StructField("loan_purpose", StringType(), True),
            StructField("original_loan_term", IntegerType(), True),
            StructField("num_borrowers", IntegerType(), True),
            StructField("seller_name", StringType(), True),
            StructField("servicer_name", StringType(), True),
            StructField("super_conforming_flag", StringType(), True),
            StructField("pre_relief_refinance_seq", StringType(), True),
            StructField("special_eligibility_program", StringType(), True),
            StructField("relief_refinance_indicator", StringType(), True),
            StructField("property_valuation_method", StringType(), True),
            StructField("io_indicator", StringType(), True),
            StructField("mi_cancellation_indicator", StringType(), True)
        ])

        perf_schema = StructType([
            StructField("loan_sequence_number", StringType(), True),
            StructField("monthly_reporting_period", StringType(), True),
            StructField("current_actual_upb", DoubleType(), True),
            StructField("current_loan_delinquency_status", StringType(), True),
            StructField("loan_age", IntegerType(), True),
            StructField("remaining_months_to_maturity", IntegerType(), True),
            StructField("defect_settlement_date", StringType(), True),
            StructField("modification_flag", StringType(), True),
            StructField("zero_balance_code", StringType(), True),
            StructField("zero_balance_effective_date", StringType(), True),
            StructField("current_interest_rate", DoubleType(), True),
            StructField("current_non_interest_bearing_upb", DoubleType(), True),
            StructField("ddlpi", StringType(), True),
            StructField("mi_recoveries", DoubleType(), True),
            StructField("net_sale_proceeds", StringType(), True),
            StructField("non_mi_recoveries", DoubleType(), True),
            StructField("total_expenses", DoubleType(), True),
            StructField("legal_costs", DoubleType(), True),
            StructField("maintenance_preservation_costs", DoubleType(), True),
            StructField("taxes_and_insurance", DoubleType(), True),
            StructField("miscellaneous_expenses", DoubleType(), True),
            StructField("actual_loss_calculation", DoubleType(), True),
            StructField("cumulative_modification_cost", DoubleType(), True),
            StructField("interest_rate_step_indicator", StringType(), True),
            StructField("payment_deferral_flag", StringType(), True),
            StructField("estimated_ltv", StringType(), True),
            StructField("zero_balance_removal_upb", DoubleType(), True),
            StructField("delinquent_accrued_interest", DoubleType(), True),
            StructField("delinquency_due_to_disaster", StringType(), True),
            StructField("borrower_assistance_status_code", StringType(), True),
            StructField("current_month_modification_cost", DoubleType(), True),
            StructField("interest_bearing_upb", DoubleType(), True)
        ])

        data_dir = Path("data")
        quarters = ["2015Q1", "2015Q2", "2015Q3", "2015Q4"]

        print("Loading origination data...")
        orig_dfs = []
        for q in quarters:
            file_path = data_dir / f"historical_data_{q}.txt"
            if file_path.exists():
                df = spark.read \
                    .option("delimiter", "|") \
                    .option("nullValue", "") \
                    .schema(orig_schema) \
                    .csv(str(file_path))
                orig_dfs.append(df)
                print(f"  {q}: {df.count():,} loans")

        if not orig_dfs:
            raise FileNotFoundError("No origination files found")

        orig_df = orig_dfs[0]
        for df in orig_dfs[1:]:
            orig_df = orig_df.unionByName(df)

        print("Loading performance data...")
        perf_dfs = []
        for q in quarters:
            file_path = data_dir / f"historical_data_time_{q}.txt"
            if file_path.exists():
                df = spark.read \
                    .option("delimiter", "|") \
                    .option("nullValue", "") \
                    .schema(perf_schema) \
                    .csv(str(file_path))
                perf_dfs.append(df)
                print(f"  {q}: loaded")

        if not perf_dfs:
            raise FileNotFoundError("No performance files found")

        perf_df = perf_dfs[0]
        for df in perf_dfs[1:]:
            perf_df = perf_df.unionByName(df)

        # ra = reo acquisition, map to 99 so it sorts as worst state
        def safe_delinq_int(c):
            return when(c == "RA", 99).otherwise(c.cast("int"))

        perf_df = perf_df.withColumn(
            "delinq_int", safe_delinq_int(col("current_loan_delinquency_status"))
        ).withColumn(
            "period_int", col("monthly_reporting_period").cast("int")
        ).withColumn(
            "mod_onset", when(col("modification_flag") == "Y", 1).otherwise(0)
        ).withColumn(
            "seriously_delinquent", when(col("delinq_int") >= 2, 1).otherwise(0)
        )

        w_loan = Window.partitionBy("loan_sequence_number")

        perf_df = perf_df \
            .withColumn(
                "first_mod_period",
                spark_min(
                    when(col("mod_onset") == 1, col("period_int"))
                ).over(w_loan)
            ) \
            .withColumn(
                "peak_delinq",
                spark_max("delinq_int").over(w_loan)
            ) \
            .withColumn(
                "peak_delinq_period",
                spark_min(
                    when(col("delinq_int") == col("peak_delinq"), col("period_int"))
                ).over(w_loan)
            )

        perf_agg = perf_df \
            .groupBy("loan_sequence_number") \
            .agg(
                spark_max("mod_onset").alias("ever_treated"),
                spark_max(
                    when(
                        (col("mod_onset") == 1) &
                        (col("period_int") < col("peak_delinq_period")),
                        1
                    ).otherwise(0)
                ).alias("treated_before_peak"),
                # ever_defaulted: foreclosure proxy via third party sale, short sale, reo
                spark_max(
                    when(col("zero_balance_code").isin(["02", "03", "09"]), 1).otherwise(0)
                ).alias("ever_defaulted"),
                spark_max(
                    when(safe_delinq_int(col("current_loan_delinquency_status")) >= 2, 1)
                    .otherwise(0)
                ).alias("ever_seriously_delinquent"),
                spark_max(
                    when(col("zero_balance_code") == "01", 1).otherwise(0)
                ).alias("ever_prepay"),
                count("*").alias("num_months_observed"),
                avg("current_actual_upb").alias("avg_upb"),
                spark_max("peak_delinq").alias("max_delinquency"),
                spark_sum("seriously_delinquent").alias("delinquency_duration"),
                spark_min("first_mod_period").alias("first_mod_period_abs"),
                spark_min("period_int").alias("first_period_abs")
            ) \
            .withColumn(
                "months_to_mod",
                when(
                    col("first_mod_period_abs").isNotNull(),
                    col("first_mod_period_abs") - col("first_period_abs")
                ).otherwise(None)
            )

        # join origination with aggregated performance, drop loans with < 6 months observed
        causal_df = orig_df \
            .join(perf_agg, on="loan_sequence_number", how="inner") \
            .filter(col("num_months_observed") >= 6)

        # risk feature engineering
        causal_df = causal_df \
            .withColumn("high_ltv", when(col("original_ltv") > 80, 1).otherwise(0)) \
            .withColumn("high_dti", when(col("original_dti") > 43, 1).otherwise(0)) \
            .withColumn("low_credit_score", when(col("credit_score") < 620, 1).otherwise(0)) \
            .withColumn("investment_property",
                        when(col("occupancy_status") == "I", 1).otherwise(0)) \
            .withColumn("cash_out_refi",
                        when(col("loan_purpose") == "C", 1).otherwise(0)) \
            .withColumn("risk_score",
                col("high_ltv") * 0.25 + col("high_dti") * 0.25 +
                col("low_credit_score") * 0.3 + col("investment_property") * 0.1 +
                col("cash_out_refi") * 0.1
            )

        causal_df = causal_df.repartition("property_state")

        pdf = causal_df.toPandas()
        output_path = data_dir / "freddie_mac_causal_ready.parquet"
        pdf.to_parquet(output_path, index=False)

        print(f"\nETL complete. Saved to {output_path}")
        print(f"Total loans: {len(pdf):,}")
        print(f"ever_treated rate: {pdf['ever_treated'].mean():.2%}")
        at_risk = (pdf["ever_seriously_delinquent"] == 1) | (pdf["ever_treated"] == 1)
        print(f"At-risk loans: {at_risk.sum():,}")
        print(f"ever_treated in at-risk: {pdf.loc[at_risk, 'ever_treated'].mean():.2%}")
        print(f"ever_defaulted in at-risk: {pdf.loc[at_risk, 'ever_defaulted'].mean():.2%}")

    except Exception as e:
        print(f"Error during ETL: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        spark.stop()

if __name__ == "__main__":
    main()