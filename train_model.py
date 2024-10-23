# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv('C:\Users\noori\retail-sector')

# Handle missing values (fill missing cluster_ids or drop rows)
data['cluster_id'].fillna('unknown', inplace=True)

# Encode categorical variables
label_encoders = {}
for column in ['cluster_id', 'hierarchy1_id', 'hierarchy2_id', 'hierarchy3_id', 'hierarchy4_id', 'hierarchy5_id']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features and Target
X = data[['product_length', 'product_depth', 'product_width', 'hierarchy1_id', 'hierarchy2_id', 'hierarchy3_id', 'hierarchy4_id', 'hierarchy5_id']]
y = data['cluster_id']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


