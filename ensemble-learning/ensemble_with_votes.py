import argparse
import os
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from sklearn.base import clone
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm
from joblib import Parallel, delayed

# -----------------------------
# 设计特征
# -----------------------------
def extract_struct_features(df: pd.DataFrame) -> np.ndarray:
    """
    构建结构化特征，从以下字段构造：
    overall
    votes_up
    votes_all
    reviewText
    """
    text = df["reviewText"].fillna("").astype(str)

    len_chars = text.str.len().astype(float) # 评论字符长度
    words = text.str.split()
    len_words = words.str.len().fillna(0).astype(float) # 单词数
    avg_word_len = (len_chars / (len_words + 1.0)).astype(float) # 平均单词长度

    num_exclaim = text.str.count("!").astype(float) # 感叹号数量
    num_question = text.str.count(r"\?").astype(float) # 问号数量

    # uppercase ratio
    def upper_ratio(s: str) -> float:
        if not s:
            return 0.0
        up = sum(1 for c in s if c.isupper())
        return up / (len(s) + 1.0)

    ratio_upper = text.apply(upper_ratio).astype(float) # 大写字母比例

    votes_up = df["votes_up"].fillna(0).astype(float) if "votes_up" in df.columns else pd.Series(0.0, index=df.index)
    votes_all = df["votes_all"].fillna(0).astype(float) if "votes_all" in df.columns else pd.Series(0.0, index=df.index) # 总投票数

    helpful_ratio = (votes_up + 1.0) / (votes_all + 2.0) # 有帮助投票比例

    overall = df["overall"].fillna(df["overall"].median()).astype(float) # 星级

    # 堆叠所有特征，这些特征对 质量！=情感 的分类任务有帮助
    feats = np.vstack([
        len_chars.values,
        len_words.values,
        avg_word_len.values,
        num_exclaim.values,
        num_question.values,
        ratio_upper.values,
        votes_all.values,
        helpful_ratio.values,
        overall.values,
    ]).T

    return feats


def build_features(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        max_word_features: int = 30000,
        max_char_features: int = 20000,
):
    """
    特征拼接：
    - word TF-IDF
    - char TF-IDF
    - 结构化特征（数值型）

    在特征构建过程中，仅在训练集上拟合 TF-IDF 向量器，以避免测试数据泄漏。
    随后使用相同的词典和 IDF 权重分别对训练集和测试集进行特征转换，从而保证二者处于一致的特征空间。
    在此基础上，将文本特征与从评论元数据中提取的结构化特征进行拼接，形成最终的模型输入。
    由于文本特征维度较高且高度稀疏，最终特征矩阵采用稀疏表示，以提升存储与计算效率。
    """
    pbar = tqdm(total=4, desc="Building features")

    train_text = train_df["reviewText"].fillna("").astype(str)
    test_text = test_df["reviewText"].fillna("").astype(str)

    word_vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        max_features=max_word_features,
    )
    Xw_tr = word_vec.fit_transform(train_text)
    Xw_te = word_vec.transform(test_text)
    pbar.update(1)


    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=max_char_features,
    )
    Xc_tr = char_vec.fit_transform(train_text)
    Xc_te = char_vec.transform(test_text)
    pbar.update(1)

    # structured feats -> scale (important for SVM)
    F_tr = extract_struct_features(train_df)
    F_te = extract_struct_features(test_df)

    scaler = StandardScaler(with_mean=False)  # safe for sparse-like usage
    F_tr_s = scaler.fit_transform(F_tr)
    F_te_s = scaler.transform(F_te)
    pbar.update(1)

    # 返回只存非零值的稀疏矩阵格式
    X_tr = hstack([Xw_tr, Xc_tr, csr_matrix(F_tr_s)], format="csr")
    X_te = hstack([Xw_te, Xc_te, csr_matrix(F_te_s)], format="csr")
    pbar.update(1)
    pbar.close()

    return X_tr, X_te, word_vec, char_vec, scaler


# -----------------------------
# 手写 Bagging
# -----------------------------
class BaggingEnsemble:
    def __init__(self, base_estimator, n_estimators=50, random_state=42, desc="Bagging training"):
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.rs = np.random.RandomState(random_state)
        self.models_ = []
        self.desc = desc

    def fit(self, X, y):
        n = X.shape[0]
        D = np.ones(n, dtype=float) / n
        self.models_ = []
        for _ in tqdm(
                    range(self.n_estimators),
                    desc=self.desc,
                    leave=False
            ):
            idx = self.rs.randint(0, n, size=n)
            model = clone(self.base_estimator)
            model.fit(X, y, sample_weight=D) # 用sample weight， 避免巨大稀疏矩阵拷贝
            self.models_.append(model)
        return self

    def predict_score(self, X):
        """
        Continuous scores for AUC.
        - DecisionTree: predict_proba[:, 1]
        - LinearSVC: decision_function
        """
        scores = []
        for m in self.models_:
            if hasattr(m, "predict_proba"):
                s = m.predict_proba(X)[:, 1]
            else:
                s = m.decision_function(X)
            scores.append(s)
        return np.mean(np.vstack(scores), axis=0)


# -----------------------------
# 手写 AdaBoost.M1
# -----------------------------
class AdaBoostM1:
    def __init__(self, base_estimator, n_estimators=100, random_state=42, desc="AdaBoost training"):
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.rs = np.random.RandomState(random_state)
        self.models_ = []
        self.alphas_ = []
        self.desc = desc

    def fit(self, X, y):
        """
        y expected in {0,1}. base estimator must support sample_weight.
        """
        n = X.shape[0]
        D = np.ones(n, dtype=float) / n
        self.models_, self.alphas_ = [], []

        for _ in tqdm(
                range(self.n_estimators),
                desc=self.desc,
                leave=False
            ):
            model = clone(self.base_estimator)
            model.fit(X, y, sample_weight=D)

            y_pred = model.predict(X)
            err = float(np.sum(D * (y_pred != y)))

            if err >= 0.5:
                continue

            err = max(err, 1e-12)
            alpha = np.log((1.0 - err) / err)

            D *= np.exp(alpha * (y_pred != y))
            D /= D.sum()

            self.models_.append(model)
            self.alphas_.append(alpha)

            if err <= 1e-10:
                break

        return self

    def predict_score(self, X):
        """
        Return continuous score F(x) = sum alpha_t * h_t(x) where h_t in {-1,+1}.
        Use F(x) directly for AUC.
        """
        F = np.zeros(X.shape[0], dtype=float)
        for alpha, m in zip(self.alphas_, self.models_):
            pred = m.predict(X)  # {0,1}
            pred_signed = 2.0 * pred - 1.0
            F += alpha * pred_signed
        return F


# -----------------------------
# Utilities
# -----------------------------
def read_csv_auto(path: str) -> pd.DataFrame:
    # try tab first then comma (your example is tab-separated but says csv)
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(path)  # fallback comma
    except Exception:
        df = pd.read_csv(path)
    return df


def auc_on_group_split(X, y, groups, model, val_ratio=0.15, seed=42):
    """
    使用分组切分（GroupShuffleSplit）做验证：
    - 保证同一个 group（例如 reviewerID 或 asin）不会同时出现在训练集和验证集中
    - 目的：避免“同一用户/同一商品”带来的隐式泄漏，得到更可信的验证 AUC

    参数：
    - X: 特征矩阵（稀疏）
    - y: 标签（0/1）
    - groups: 分组ID数组（长度与y一致），例如 train_df["reviewerID"] 或 train_df["asin"]
    - model: 你的手写集成模型（BaggingEnsemble / AdaBoostM1）
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_tr, X_va = X[train_idx], X[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]

    # 训练并在验证集上算 AUC
    model.fit(X_tr, y_tr)
    score_va = model.predict_score(X_va)

    # 如果某次分组切分导致验证集只有一个类别，AUC 会报错
    # 这种情况说明 val_ratio 太小或 group 分布太极端，可以调大 val_ratio 或换分组字段
    if len(np.unique(y_va)) < 2:
        raise ValueError("验证集中只有一个类别，无法计算 AUC。请增大 val_ratio 或更换分组字段。")

    auc = roc_auc_score(y_va, score_va)
    return auc, model

def save_submission(test_ids, scores, out_path):
    sub = pd.DataFrame({"Id": test_ids, "Predicted": scores})
    sub.to_csv(out_path, index=False)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv", help="train file path (has label)")
    ap.add_argument("--test", default="data/test.csv", help="test file path (has Id)")
    ap.add_argument("--groundtruth", default="data/groundTruth.csv", help="optional groundTruth csv: Id,Expected")
    ap.add_argument("--outdir", default="outputs", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train_df = read_csv_auto(args.train)
    test_df = read_csv_auto(args.test)

    # basic sanity
    needed_train = {"reviewText", "overall", "label"}
    missing = needed_train - set(train_df.columns)
    if missing:
        raise ValueError(f"Train missing columns: {missing}")

    if "Id" not in test_df.columns:
        raise ValueError("Test file must contain column: Id")

    y = train_df["label"].astype(int).values
    X_train, X_test, *_ = build_features(train_df, test_df)

    test_ids = test_df["Id"].values

    # -----------------------------
    # 选择分组字段：优先 reviewerID，其次 asin
    # 目的：做 group validation，避免“同一用户/商品”同时出现在训练与验证中造成泄漏
    # -----------------------------
    if "reviewerID" in train_df.columns:
        groups = train_df["reviewerID"].astype(str).values
        group_name = "reviewerID"
    elif "asin" in train_df.columns:
        groups = train_df["asin"].astype(str).values
        group_name = "asin"
    else:
        raise ValueError("train.csv 缺少 reviewerID/asin，无法进行 group validation。")

    print(f"\n[Group Validation] 使用分组字段: {group_name}")

    # -----------------------------
    # Define base learners
    # -----------------------------
    # Bagging + DT (deep-ish tree works well for bagging)
    base_dt_bag = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=2,
        random_state=42
    )

    # AdaBoost + DT (stump/weak tree)
    base_dt_ada = DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=10,
        random_state=42
    )

    # SVM base (weaken via C for AdaBoost if needed)
    base_svm_bag = LinearSVC(C=0.2, max_iter=8000) # 跑的过程出现警告，加大迭代次数
    base_svm_ada = LinearSVC(C=0.2, max_iter=20000)

    # -----------------------------
    # Build 4 models
    # -----------------------------
    models = {
        "Bagging_SVM": BaggingEnsemble(base_svm_bag, n_estimators=12, random_state=42, desc="Bagging + SVM"),
        "Bagging_DT": BaggingEnsemble(base_dt_bag, n_estimators=50, random_state=42, desc="Bagging + DT"),
        "AdaBoost_SVM": AdaBoostM1(base_svm_ada, n_estimators=12, random_state=42, desc="AdaBoost + SVM"),
        "AdaBoost_DT": AdaBoostM1(base_dt_ada, n_estimators=70, random_state=42, desc="AdaBoost + DT"),
    }

    results = []

    for name, model in tqdm(
            models.items(),
            desc="Training ensemble models"
        ):
        auc, fitted_model = auc_on_group_split(X_train, y,groups, model, val_ratio=0.15, seed=42)
        results.append((name, auc))

        # fit on full training and predict test
        fitted_model.fit(X_train, y)
        test_scores = fitted_model.predict_score(X_test)

        out_path = os.path.join(args.outdir, f"submission_{name}.csv")
        save_submission(test_ids, test_scores, out_path)
        print(f"[{name}] valid AUC = {auc:.4f} | saved: {out_path}")

    # print summary
    results.sort(key=lambda x: x[1], reverse=True)
    print("\n=== Validation AUC Summary (higher is better) ===")
    for name, auc in results:
        print(f"{name:12s}: {auc:.4f}")

    # optional: evaluate on groundTruth (if provided)
    if args.groundtruth:
        gt = pd.read_csv(args.groundtruth)
        if not {"Id", "Expected"} <= set(gt.columns):
            raise ValueError("groundTruth must have columns: Id, Expected")

        gt_map = dict(zip(gt["Id"].values, gt["Expected"].values))
        # align y_true with test_ids
        mask = np.array([i in gt_map for i in test_ids])
        if mask.sum() == 0:
            print("\n[groundTruth] No matching Id found between test and groundTruth.")
            return

        y_true = np.array([gt_map[i] for i in test_ids[mask]]).astype(int)

        print("\n=== Test AUC using groundTruth (if test is labeled) ===")
        for name in models.keys():
            sub_path = os.path.join(args.outdir, f"submission_{name}.csv")
            pred = pd.read_csv(sub_path)
            pred_map = dict(zip(pred["Id"].values, pred["Predicted"].values))
            y_score = np.array([pred_map[i] for i in test_ids[mask]])
            auc_test = roc_auc_score(y_true, y_score)
            print(f"{name:12s}: {auc_test:.4f}")

if __name__ == "__main__":
    main()
