"""Behavioral prediction study — H&M regime analysis."""

SEED = 42
T_TRAIN_CUTOFF = "2020-07-22"
T_TEST_CUTOFF = "2020-08-22"
LABEL_WINDOW_DAYS = 30
SAMPLE_N = 50_000
LLM_SAMPLE_N = 5_000
ACTIVITY_BUCKETS = [(1, 1), (2, 5), (6, 20), (21, 100), (101, 10**9)]
