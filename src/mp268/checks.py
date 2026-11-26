import yaml
import pandas as pd


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_unit_mixup(df: pd.DataFrame, numeric_cols: list[str], group_col: str = "zone") -> dict:
    findings = {"present": False, "details": {}}

    for col in numeric_cols:
        stats = df.groupby(group_col)[col].agg(["min", "mean", "max"])
        overall_median_mean = stats["mean"].median()

        for zone_val, row in stats.iterrows():
            if overall_median_mean > 0:
                ratio = row["mean"] / overall_median_mean
                for factor in [10, 12, 60, 100, 1000]:
                    if 0.8 * factor <= ratio <= 1.2 * factor:
                        findings["present"] = True
                        findings["details"] = {
                            "column": col,
                            "group_column": group_col,
                            "affected_group": zone_val,
                            "factor": factor,
                            "zone_stats": stats.round(2).to_dict(orient="index"),
                        }
                        break

    return findings


def check_stuck_sensor(df: pd.DataFrame, numeric_cols: list[str], min_run_days: int = 7) -> dict:
    df_sorted = df.drop_duplicates().sort_values("date").reset_index(drop=True)
    findings = {"present": False, "longest_runs": {}}

    for col in numeric_cols:
        run_ids = (df_sorted[col] != df_sorted[col].shift()).cumsum()
        runs = df_sorted.groupby(run_ids).agg(
            value=(col, "first"),
            row_count=(col, "size"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        runs["span_days"] = (
            pd.to_datetime(runs["end_date"]) - pd.to_datetime(runs["start_date"])
        ).dt.days + 1

        top_run = runs.sort_values(["span_days", "row_count"], ascending=False).iloc[0]
        findings["longest_runs"][col] = {
            "value": float(top_run["value"]),
            "row_count": int(top_run["row_count"]),
            "span_days": int(top_run["span_days"]),
            "start_date": str(top_run["start_date"]),
            "end_date": str(top_run["end_date"]),
        }

        if top_run["span_days"] >= min_run_days:
            findings["present"] = True
            findings["column"] = col
            findings["start_date"] = str(top_run["start_date"])
            findings["end_date"] = str(top_run["end_date"])

    return findings


if __name__ == "__main__":
    cfg = load_params()
    train_df = pd.read_csv(cfg["paths"]["raw_train"])
    num_cols = cfg["features"]["numeric"]

    print(yaml.safe_dump(check_unit_mixup(train_df, num_cols), sort_keys=False))
    print(yaml.safe_dump(check_stuck_sensor(train_df, num_cols), sort_keys=False))