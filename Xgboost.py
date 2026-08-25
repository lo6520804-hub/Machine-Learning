import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

X = df[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]]
y = df["Survived"]

hanh_khach_moi = pd.DataFrame([{
    "Pclass": 1,      
    "Age": 19,        
    "SibSp": 2,       
    "Parch": 0,        
    "Fare": 100,       
    "Sex": 0,       
    "Embarked": 1      
}])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33
)

model_xgb = XGBClassifier(random_state=33)
model_xgb.fit(X_train, y_train)

du_doan = model_xgb.predict(hanh_khach_moi)
xac_suat = model_xgb.predict_proba(hanh_khach_moi)

y_pred_xgb = model_xgb.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
precision = precision_score(y_test, y_pred_xgb)
recall = recall_score(y_test, y_pred_xgb)
f1 = f1_score(y_test, y_pred_xgb)

print("=== XGBoost ===")
print("Độ chính xác:", acc_xgb)
print("Confusion Matrix:\n", cm_xgb)

print("Dự đoán (0=chết, 1=sống):", du_doan[0])
print("Xác suất [chết, sống]:", xac_suat[0])

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)