import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

df = pd.read_excel('IMU_Fall_Data.xlsx', sheet_name='Sheet1')


def parse_series(cell):
    parts = str(cell).split(',')
    return np.array([float(x) for x in parts if x.strip() != ''], dtype=np.float32)


features, labels = [], []
for _, row in df.iterrows():
    feats = {}
    for axis in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
        arr = parse_series(row[axis])
        feats[f'{axis}_mean'] = arr.mean()
        feats[f'{axis}_std'] = arr.std()
        feats[f'{axis}_min'] = arr.min()
        feats[f'{axis}_max'] = arr.max()
    features.append(feats)
    labels.append(row['classification'])

X = pd.DataFrame(features, dtype=np.float32)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)

best_pipeline = grid.best_estimator_
y_pred = best_pipeline.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

n_features = X_train.shape[1]
initial_type = [('float_input', FloatTensorType([None, n_features]))]

onnx_model = convert_sklearn(best_pipeline, initial_types=initial_type)
with open("fall_detector.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved as fall_detector.onnx")
