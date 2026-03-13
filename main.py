import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("fertilizer_dataset_500_rows.csv")

# Label Encoding
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

data["Soil"] = le_soil.fit_transform(data["Soil"])
data["Crop"] = le_crop.fit_transform(data["Crop"])
data["Fertilizer"] = le_fert.fit_transform(data["Fertilizer"])

# Features and Target
X = data.drop(["Fertilizer","Remark"], axis=1)
y = data["Fertilizer"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))

# Save Model
pickle.dump(
    (model, le_soil, le_crop, le_fert),
    open("fertilizer_model.pkl","wb")
)

print("Model saved successfully!")