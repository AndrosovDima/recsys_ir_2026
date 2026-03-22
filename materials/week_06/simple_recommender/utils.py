from datetime import date

import polars as pl
import pandas as pd
from rectools.metrics import (
    Precision,
    NDCG,
    Recall,
    MAP,
    calc_metrics,
)

FINAL_TOP_K = 30


def calc_anp_print_metrics(recommendations: pd.DataFrame, interactions: pd.DataFrame, top_k: int = FINAL_TOP_K):
    metrics = {
        "recall": Recall(k=top_k),
        "precision": Precision(k=top_k),
        "ndcg": NDCG(k=top_k),
        "map_": MAP(k=top_k),
    }
    metrics = calc_metrics(
        metrics=metrics,
        reco=recommendations,
        interactions=interactions,
    )
    print(metrics)


def prepare_test_data(user_actions_full: pl.DataFrame, test_start: date) -> pl.DataFrame:
    sample_users = pl.read_parquet("data/sample_users.parquet").limit(5)
    return sample_users
