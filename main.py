import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("fertilizer_recommendation_dataset.csv")

le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

data["Soil"] = le_soil.fit_transform(data["Soil"])
data["Crop"] = le_crop.fit_transform(data["Crop"])
data["Fertilizer"] = le_fert.fit_transform(data["Fertilizer"])

X = data.drop(["Fertilizer", "Remark"], axis=1)
y = data["Fertilizer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)
pickle.dump((model, le_soil, le_crop, le_fert),
            open("fertilizer_model.pkl", "wb"))

print("Model trained successfully")