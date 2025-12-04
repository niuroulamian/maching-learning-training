import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# === 读取数据 ===
data_df = pd.read_csv('./cwurData.csv')  # 读入 csv 文件为 pandas 的 DataFrame

data_df = data_df.dropna()  # 舍去包含 NaN 的 row
len(data_df)

# === 特征与目标 （这次去掉 broad_impact ） ===
numeric_features = [
    "quality_of_education",
    "alumni_employment",
    "quality_of_faculty",
    "publications",
    "influence",
    "citations",
    "broad_impact",
    "patents"
]

categorical_features = ["region"]

X = data_df[numeric_features + categorical_features]
y = data_df["score"]

# === One-hot Encoding ===
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop="first"), categorical_features)
    ],
    remainder='passthrough'  # keep numeric features
)

# ---------------------
# 建立线性回归流水线
# ---------------------
model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('reg', LinearRegression())
])

# ---------------------
# Train-test split（与之前保持 8:2）
# ---------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------
# 训练
# ---------------------
model.fit(X_train, y_train)

# ---------------------
# 预测 & RMSE
# ---------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE with region:", rmse)

# ---------------------
# 提取系数（含地区）
# ---------------------
ohe = model.named_steps["preprocess"].named_transformers_["cat"]
region_encoded_names = ohe.get_feature_names_out(categorical_features)

feature_names = list(region_encoded_names) + numeric_features
coef = model.named_steps["reg"].coef_

coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
print("\nCoefficients including region:")
print(coef_df)
