import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# === 1. 读取数据 ===
data_df = pd.read_csv('./cwurData.csv')  # 读入 csv 文件为 pandas 的 DataFrame

data_df = data_df.dropna()  # 舍去包含 NaN 的 row
len(data_df)

# === 2. 特征与目标 ===
feature_cols = ['quality_of_faculty', 'publications', 'citations', 'alumni_employment',
                'influence', 'quality_of_education', 'broad_impact', 'patents']
x = data_df[feature_cols]
y = data_df['score']

# === 3. 训练集/测试集 8:2 随机划分 ===
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# === 4. 训练线性回归模型 ===
model = LinearRegression()
model.fit(x_train, y_train)

# === 5. 预测并计算 RMSE ===
y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test RMSE:", rmse)

# === 6. 输出模型系数 ===
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": model.coef_
}).sort_values(by="coefficient", ascending=False)

print("\nCoefficients:")
print(coef_df)

vif_data = pd.DataFrame()
vif_data["feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
print("\nVariance Inflation Factors (VIF):")
print(vif_data[["feature", "VIF"]])

plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red', linestyle='--'
)

plt.xlabel("Actual Score")
plt.ylabel("Predicted Score (Linear Regression)")
plt.title("Predicted vs Actual (Linear Regression)")
plt.show()