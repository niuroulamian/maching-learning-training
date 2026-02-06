import numpy as np
import pandas as pd

def build_user_features(level_seq: pd.DataFrame) -> pd.DataFrame:
    """
    从事件级日志构建用户级特征
    """
    df = level_seq.copy()
    df = df.sort_values(["user_id", "time"])

    agg = df.groupby("user_id").agg(
        n_attempts=("level_id", "count"),
        n_levels=("level_id", "nunique"),

        success_cnt=("f_success", "sum"),
        fail_cnt=("f_success", lambda x: (1 - x).sum()),
        success_rate=("f_success", "mean"),

        avg_duration=("f_duration", "mean"),
        std_duration=("f_duration", "std"),
        max_duration=("f_duration", "max"),

        avg_reststep=("f_reststep", "mean"),
        help_rate=("f_help", "mean"),

        first_time=("time", "min"),
        last_time=("time", "max"),
    ).reset_index()

    # 衍生特征
    agg["fail_success_ratio"] = agg["fail_cnt"] / (agg["success_cnt"] + 1)
    agg["last_time"] = pd.to_datetime(
        agg["last_time"],
    )
    agg["first_time"] = pd.to_datetime(
        agg["first_time"],
    )
    agg["active_span"] = agg["last_time"] - agg["first_time"]

    agg["last_time"] = agg["last_time"].to_numpy(dtype="float32")
    agg["first_time"] = agg["first_time"].to_numpy(dtype="float32")
    agg["active_span"] = agg["active_span"].to_numpy(dtype="float32")

    return agg

def longest_fail_streak(x):
    max_streak = streak = 0
    for v in x:
        if v == 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def build_streak_features(level_seq):
    df = level_seq.sort_values(["user_id", "time"])

    streak = df.groupby("user_id")["f_success"].apply(longest_fail_streak)
    streak = streak.reset_index().rename(columns={"f_success": "max_fail_streak"})
    return streak

def build_level_meta_features(level_seq, level_meta):
    df = level_seq.merge(level_meta, on="level_id", how="left")

    agg = df.groupby("user_id").agg(
        avg_level_passrate=("f_avg_passrate", "mean"),
        max_level_passrate=("f_avg_passrate", "max"),

        avg_level_difficulty=("f_avg_retrytimes", "mean"),
        max_level_difficulty=("f_avg_retrytimes", "max"),

        avg_level_duration=("f_avg_duration", "mean"),
    ).reset_index()

    return agg

def build_features(level_seq, level_meta):
    f1 = build_user_features(level_seq)
    f2 = build_streak_features(level_seq)
    f3 = build_level_meta_features(level_seq, level_meta)

    feats = f1.merge(f2, on="user_id", how="left") \
        .merge(f3, on="user_id", how="left")

    return feats
