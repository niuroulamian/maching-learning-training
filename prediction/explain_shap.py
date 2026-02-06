import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from load_data import load_basic_splits, load_sequence_data
from features import build_features


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    data_dir = "data"
    out_dir = "artifacts/shap"
    ensure_dir(out_dir)

    # 1) 读数据 + 构建特征
    train_df, dev_df, _ = load_basic_splits(data_dir)
    level_seq, level_meta = load_sequence_data(data_dir)
    feats = build_features(level_seq, level_meta)

    train = train_df.merge(feats, on="user_id", how="left")
    dev = dev_df.merge(feats, on="user_id", how="left")

    # 2) 准备训练数据
    X_train = train.drop(columns=["user_id", "label"])
    y_train = train["label"]

    X_dev = dev.drop(columns=["user_id", "label"])
    y_dev = dev["label"]

    # 缺失值处理（LightGBM可原生处理NaN，但这里可选填充）
    # X_train = X_train.fillna(-1)
    # X_dev = X_dev.fillna(-1)

    # 3) 训练一个模型（示例参数，可用你已调好的）
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
    )

    # 验证AUC（报告里也要写）
    dev_pred = model.predict_proba(X_dev)[:, 1]
    auc = roc_auc_score(y_dev, dev_pred)
    print(f"[DEV] AUC = {auc:.4f}")

    # 4) 计算 SHAP
    # 对树模型用 TreeExplainer 最合适
    explainer = shap.TreeExplainer(model)

    # 如果 dev 很大，可以抽样以提升速度（建议 2000~5000）
    sample_n = min(3000, len(X_dev))
    X_explain = X_dev.sample(sample_n, random_state=42)

    exp = explainer(X_explain)
    # exp.values 可能是 (n, m) 或 see below
    vals = exp.values
    if vals.ndim == 3:   # (n, classes, m) 或 (n, m, classes) 视版本
        # 常见是 (n, m, 2) -> 取正类
        shap_values_pos = vals[:, :, 1]
    else:
        shap_values_pos = vals

    # 5) 全局重要性图（beeswarm：看方向 + 分布）
    plt.figure()
    shap.summary_plot(shap_values_pos, X_explain, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary_beeswarm.png"), dpi=200)
    plt.close()

    # 6) 全局重要性（bar：只看强弱，不看方向）
    plt.figure()
    shap.summary_plot(shap_values_pos, X_explain, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_summary_bar.png"), dpi=200)
    plt.close()

    # 7) 单特征依赖图（dependence plot）
    # 挑一个你最关心的特征名（必须在列里存在）
    # 例如：max_fail_streak / success_rate / last_time 等
    target_feature = "max_fail_streak"
    if target_feature in X_explain.columns:
        plt.figure()
        shap.dependence_plot(
            target_feature,
            shap_values_pos,
            X_explain,
            show=False,
            interaction_index="auto"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_dependence_{target_feature}.png"), dpi=200)
        plt.close()
    else:
        print(f"Feature '{target_feature}' not found; skip dependence plot.")

    # 8) 个体解释（force plot / waterfall）
    # force_plot 适合在 notebook；脚本里更推荐 waterfall plot（保存成图片）
    # 取一个“高风险”用户样本
    idx = np.argmax(model.predict_proba(X_explain)[:, 1])
    x_row = X_explain.iloc[idx:idx+1]

    # shap.Explanation 形式（新版本 shap 推荐）
    exp = explainer(x_row)

    plt.figure()
    shap.plots.waterfall(exp[0], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_waterfall_one_user.png"), dpi=200)
    plt.close()

    print(f"Saved SHAP figures to: {out_dir}")


if __name__ == "__main__":
    main()
