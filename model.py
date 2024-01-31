import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Remove rows with zero values in specific columns
df = df[(df.BloodPressure != 0) & (df.BMI != 0) & (df.Glucose != 0) & (df.SkinThickness != 0)& (df.Age != 0) & (df.Insulin != 0)]
df.to_csv('cleaned.csv')
# Define features (X) and target variable (y)
features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
X = df[features]
y = df["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=df.Outcome, random_state=0)

# Choose a classifier (Logistic Regression in this case)
classifier = LogisticRegression(max_iter=1000)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
from sklearn.model_selection import cross_val_score

# Perform cross-validation
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())

# Save the trained model to a file
with open("model.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)