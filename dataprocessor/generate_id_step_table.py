from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STEP_MINUTES = 5
SIM_START_HOUR = 6
SIM_END_HOUR = 24
TOTAL_STEPS = (SIM_END_HOUR - SIM_START_HOUR) * (60 // STEP_MINUTES)  # 216


def build_id_step_table(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(
        input_csv,
        header=None,
        names=["id", "lat", "lon", "timestamp"],
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Build 06:00 baseline per day and floor-align by 5-minute steps.
    day_start = df["timestamp"].dt.normalize() + pd.to_timedelta(SIM_START_HOUR, unit="h")
    step = ((df["timestamp"] - day_start).dt.total_seconds() // (STEP_MINUTES * 60)).astype("int64")

    df = df.assign(step=step)
    df = df[(df["step"] >= 0) & (df["step"] < TOTAL_STEPS)]

    first_appearance = (
        df.sort_values(["id", "timestamp"])
        .groupby("id", as_index=False)
        .first()[["id", "step"]]
        .sort_values("id")
        .reset_index(drop=True)
    )

    first_appearance["id"] = first_appearance["id"].astype("int64")
    first_appearance["step"] = first_appearance["step"].astype("int64")
    return first_appearance


def main() -> None:
    input_files = [f'dataset/201408{i}.csv' for i in range(18, 23)]
    i = 18
    for input_csv in input_files:
        table = build_id_step_table(input_csv)
        table.to_csv(f"dataset/table_201408{i}.csv", index=False)
        i = i + 1


if __name__ == "__main__":
    main()
