import pandas as pd
import lightgbm as lgb

from load_data import load_basic_splits, load_sequence_data
from features import build_features


def main():
    data_dir = "data"

    train_df, _, test_df = load_basic_splits(data_dir)
    level_seq, level_meta = load_sequence_data(data_dir)

    feats = build_features(level_seq, level_meta)

    train = train_df.merge(feats, on="user_id", how="left")
    test  = test_df.merge(feats, on="user_id", how="left")

    X_train = train.drop(columns=["user_id", "label"])
    y_train = train["label"]
    X_test  = test.drop(columns=["user_id"])

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31
    )

    model.fit(X_train, y_train)

    test["churn_prob"] = model.predict_proba(X_test)[:, 1]
    test[["user_id", "churn_prob"]].to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
