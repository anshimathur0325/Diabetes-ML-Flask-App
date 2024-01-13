import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv("diabetes.csv")

# Remove rows with zero values in specific columns
df = df[(df.BloodPressure != 0) & (df.BMI != 0) & (df.Glucose != 0)]

# Define features (X) and target variable (y)
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
X = df[features]
y = df["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=df.Outcome, random_state=0)

# Feature scaling (not always necessary for Random Forest, but can be done for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Choose a classifier (Random Forest in this case)
classifier = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=0)

# Train the classifier
classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test_scaled)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Save the trained model to a file
with open("model.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)