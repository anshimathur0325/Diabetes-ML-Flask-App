import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv("diabetesv2.csv")

# Remove rows with zero values in specific columns
df2 = df.drop(columns=["Fruits","Veggies","NoDocbcCost"])
df2.to_csv("cleaned2.csv", index=False)
# Define features (X) and target variable (y)
features = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack","PhysActivity","HvyAlcoholConsump","AnyHealthcare","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"]
X = df[features]
X = X.values
y = df["Diabetes_012"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Choose a classifier (Logistic Regression in this case)
classifier = LogisticRegression(max_iter=10000)

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