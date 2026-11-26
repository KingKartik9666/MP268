import os
import yaml
import numpy as np
import pandas as pd


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_dataframe(df: pd.DataFrame, cfg: dict, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()
    clean_cfg = cfg["cleaning"]

    leak_col = cfg["features"]["leaking_column"]
    df = df.drop(columns=[leak_col], errors="ignore")

    if is_train:
        df = df.drop_duplicates().reset_index(drop=True)
        zone_mask = df["zone"] == clean_cfg["unit_mixup_zone"]
        col = clean_cfg["unit_mixup_column"]
        df.loc[zone_mask, col] = df.loc[zone_mask, col] / clean_cfg["unit_mixup_factor"]

    df.loc[df["staff"] == clean_cfg["staff_sentinel"], "staff"] = np.nan

    own_map = clean_cfg["ownership_map"]
    df["ownership"] = (
        df["ownership"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(own_map)
    )

    return df


def main():
    cfg = load_params()
    raw_train = pd.read_csv(cfg["paths"]["raw_train"])
    cleaned_train = clean_dataframe(raw_train, cfg, is_train=True)

    out_path = cfg["paths"]["clean_train"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cleaned_train.to_csv(out_path, index=False)
    print(f"Saved cleaned training data: {cleaned_train.shape} -> {out_path}")


if __name__ == "__main__":
    main()