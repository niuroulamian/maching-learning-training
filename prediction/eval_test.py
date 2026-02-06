import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    # 1. 读你的预测结果
    # 例如 predict.py 生成的 submission.csv
    pred = pd.read_csv("submission.csv")
    # 格式: user_id, churn_prob

    # 2. 读 ground truth
    gt = pd.read_csv("GroundTruth.csv")
    # ID, Label

    # 3. 对齐 ID
    df = pred.merge(
        gt,
        left_on="user_id",
        right_on="ID",
        how="inner"
    )

    assert len(df) > 0, "No matched IDs between prediction and groundTruth!"

    # 4. 计算 AUC
    auc = roc_auc_score(df["Label"], df["churn_prob"])
    print(f"[TEST] AUC = {auc:.4f}")

    # 5. （可选）保存对齐后的结果，方便分析
    df.to_csv("test_with_label.csv", index=False)


if __name__ == "__main__":
    main()
