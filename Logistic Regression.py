import pandas as pd                                    
from sklearn.model_selection import train_test_split     
from sklearn.linear_model import LogisticRegression     
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("titanic.csv")

df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

df["Age"] = df["Age"].fillna(df["Age"].median())

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
    X,y, test_size = 0.2, random_state= 33
)

model = LogisticRegression()
model.fit(X_train, y_train)
du_doan = model.predict(hanh_khach_moi)
xac_suat = model.predict_proba(hanh_khach_moi)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Độ chính xác (Accuracy):", acc)
print("\nConfusion Matrix:")
print(cm)
print("Dự đoán (0=chết, 1=sống):", du_doan[0])
print("Xác suất [chết, sống]:", xac_suat[0])