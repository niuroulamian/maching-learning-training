import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# === 读取数据 ===
data_df = pd.read_csv('./cwurData.csv')  # 读入 csv 文件为 pandas 的 DataFrame

data_df = data_df.dropna()  # 舍去包含 NaN 的 row
len(data_df)

# === 特征与目标 （这次去掉 broad_impact ） ===
feature_cols = ['quality_of_faculty', 'publications', 'citations', 'alumni_employment',
                'influence', 'quality_of_education', 'patents']
x = data_df[feature_cols]
y = data_df['score']

# === 标准化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# === PCA降维 ===
# 降到4个主成分：
# alumni_employment
# patents
#（publications，citations，influence）
#（quality_of_education，quality_of_faculty）
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)
print("解释方差占比:", pca.explained_variance_ratio_)
print("累计解释方差:", pca.explained_variance_ratio_.sum())

# === 训练集/测试集 8:2 随机划分 ===
x_train_pca, x_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# === 训练线性回归模型 ===
model = LinearRegression()
model.fit(x_train_pca, y_train)

# === 预测并计算 RMSE ===
y_pred = model.predict(x_test_pca)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse

# === 查看 PCA 主成分的含义 ===
loadings = pd.DataFrame(
    pca.components_,
    columns=feature_cols,
    index=[f"PC{i+1}" for i in range(pca.n_components_)]
)
loadings

plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red', linestyle='--'
)

plt.xlabel("Actual Score")
plt.ylabel("Predicted Score (PCA Regression)")
plt.title("Predicted vs Actual (PCA Linear Regression)")
plt.show()