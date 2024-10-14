#  model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data (you can replace this with patient-health dataset)
data = {
    'age': [25, 45, 35, 50, 23],
    'blood_pressure': [120, 130, 110, 140, 100],
    'cholesterol': [200, 220, 180, 250, 170],
    'is_healthy': [1, 0, 1, 0, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df[['age', 'blood_pressure', 'cholesterol']]
y = df['is_healthy']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple model (Random Forest)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Save the model
joblib.dump(clf, 'model.joblib')

