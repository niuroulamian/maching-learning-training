from sklearn.metrics import roc_auc_score

from load_data import load_basic_splits, load_sequence_data
from features import build_features
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


def main():
    data_dir = "data"

    train_df, dev_df, _ = load_basic_splits(data_dir)
    level_seq, level_meta = load_sequence_data(data_dir)

    feats = build_features(level_seq, level_meta)

    train = train_df.merge(feats, on="user_id", how="left")
    dev   = dev_df.merge(feats, on="user_id", how="left")

    X_train = train.drop(columns=["user_id", "label"])
    y_train = train["label"]

    X_dev = dev.drop(columns=["user_id", "label"])
    y_dev = dev["label"]

    model = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_dev, y_dev)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(50)
        ]
    )

    pred = model.predict_proba(X_dev)[:, 1]
    auc = roc_auc_score(y_dev, pred)
    print(f"Validation AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
