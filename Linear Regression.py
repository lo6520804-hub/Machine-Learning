from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 1. Load dữ liệu
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="MedHouseVal")

# 2. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Khởi tạo và train mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Xem các hệ số mô hình học được
print("Hệ số (w):", model.coef_)
print("Hệ số chặn (b):", model.intercept_)

# 5. Dự đoán trên tập test
y_pred = model.predict(X_test)

# 6. Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMSE:", mse)
print("R²:", r2)

# 7. So sánh thử 5 dòng đầu: dự đoán vs thực tế
print("\nDự đoán:", y_pred[:5])
print("Thực tế:", y_test[:5].values)