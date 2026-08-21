import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

hanh_khach_moi = pd.DataFrame([{
    "Pclass": 1,      
    "Age": 19,        
    "SibSp": 2,       
    "Parch": 0,        
    "Fare": 100,       
    "Sex": 0,       
    "Embarked": 1
}])

X = df[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33
)

model_tree = DecisionTreeClassifier(random_state=33)
model_tree.fit(X_train, y_train)
y_pred_tree = model_tree.predict(X_test)

du_doan = model_tree.predict(hanh_khach_moi)
xac_suat = model_tree.predict_proba(hanh_khach_moi)

acc_tree = accuracy_score(y_test, y_pred_tree)
cm_tree = confusion_matrix(y_test, y_pred_tree)

print("\n=== Decision Tree ===")
print("Độ chính xác:", acc_tree)
print("Confusion Matrix:\n", cm_tree)

print("Dự đoán (0=chết, 1=sống):", du_doan[0])
print("Xác suất [chết, sống]:", xac_suat[0])