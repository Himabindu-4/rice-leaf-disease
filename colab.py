# code for random forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset (replace 'path_to_file' with the actual file path in your Colab environment)
file_path = "/content/archive.zip"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Step 1: Data Preprocessing
# Drop unnecessary columns (e.g., 'index' and 'Patient Id')
data = data.drop(['index', 'Patient Id'], axis=1)

# Encode categorical target variable ('Level')
label_encoder = LabelEncoder()
data['Level'] = label_encoder.fit_transform(data['Level'])

# Separate features and target variable
X = data.drop('Level', axis=1)
y = data['Level']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values for SVM (Random Forest doesn't require scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Train models

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test

# code for support vector machine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset (replace 'path_to_file' with the actual file path in your Colab environment)
file_path = "/content/archive.zip"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Step 1: Data Preprocessing
# Drop unnecessary columns (e.g., 'index' and 'Patient Id')
data = data.drop(['index', 'Patient Id'], axis=1)

# Encode categorical target variable ('Level')
label_encoder = LabelEncoder()
data['Level'] = label_encoder.fit_transform(data['Level'])

# Separate features and target variable
X = data.drop('Level', axis=1)
y = data['Level']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values for SVM (Random Forest doesn't require scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Train models

# Random Forest Classifier
svm_model = supportvectormachine(random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test

# SVM Evaluation
print("\nSupport Vector Machine (SVM) Classifier:")
print(f"Accuracy: {accuracy_score(y_test, svm_predictions):.2f}")
print("Classification Report:\n", classification_report(y_test, svm_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))

# code for comparison

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset (replace 'file_path' with the actual file path in your Colab environment)
file_path = "/content/archive.zip"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Step 1: Data Preprocessing
# Drop unnecessary columns (e.g., 'index' and 'Patient Id')
data = data.drop(['index', 'Patient Id'], axis=1)

# Encode categorical target variable ('Level')
label_encoder = LabelEncoder()
data['Level'] = label_encoder.fit_transform(data['Level'])

# Separate features and target variable
X = data.drop('Level', axis=1)
y = data['Level']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values for SVM (Random Forest doesn't require scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Train models

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Support Vector Machine (SVM) Classifier
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)

# Step 3: Evaluation and Comparison

# Random Forest Evaluation
print("Random Forest Classifier:")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions):.2f}")
print("Classification Report:\n", classification_report(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))

# SVM Evaluation
print("\nSupport Vector Machine (SVM) Classifier:")
print(f"Accuracy: {accuracy_score(y_test, svm_predictions):.2f}")
print("Classification Report:\n", classification_report(y_test, svm_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))

