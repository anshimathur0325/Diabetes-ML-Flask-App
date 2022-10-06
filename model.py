import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score

df = pd.read_csv("diabetes.csv")
df = df[(df.BloodPressure != 0) & (df.BMI != 0) & (df.Glucose != 0)]

X = df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=df.Outcome, random_state=0)

classifier = LogisticRegression(max_iter=1000)

classifier.fit(X_train, y_train)

logpre = classifier.predict(X_test)
print(accuracy_score(y_test,logpre))


pickle.dump(classifier, open("model.pkl", "wb"))
