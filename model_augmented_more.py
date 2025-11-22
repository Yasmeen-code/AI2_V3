import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pickle

# -------------------------
# 1️⃣ Load Data
# -------------------------
df = pd.read_csv("augmented_more.csv")
print("Original Shape:", df.shape)

TARGET = "Health_Score"

# -------------------------
# 2️⃣ Handle Missing Values
# -------------------------
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------
# 3️⃣ Remove Duplicates
# -------------------------
df = df.drop_duplicates()

# -------------------------
# 4️⃣ Remove Outliers using IQR
# -------------------------
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove(TARGET)

for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("Shape after removing duplicates and outliers:", df.shape)

# -------------------------
# 5️⃣ Separate Features and Target
# -------------------------
X = df.drop(TARGET, axis=1)
y = df[TARGET]

categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# -------------------------
# 6️⃣ One-Hot Encoding
# -------------------------
encoder = OneHotEncoder(sparse_output=False, drop='first')
if categorical_features:
    X_cat = encoder.fit_transform(X[categorical_features])
    cat_cols = encoder.get_feature_names_out(categorical_features)
    df_cat = pd.DataFrame(X_cat, columns=cat_cols, index=X.index)
    X = pd.concat([X.drop(columns=categorical_features), df_cat], axis=1)

# -------------------------
# 7️⃣ Scale Numeric Features
# -------------------------
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# -------------------------
# 8️⃣ Save Processed Data
# -------------------------
processed_df = X.copy()
processed_df[TARGET] = y.values
processed_df.to_csv("augmented_more_processed.csv", index=False)
print("Processed data saved successfully as 'augmented_more_processed.csv'")

# -------------------------
# 9️⃣ Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -------------------------
# 🔟 Hyperparameter Tuning for Random Forest
# -------------------------
param_grid = {
    'n_estimators': [300, 400],
    'max_depth': [15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# -------------------------
# 1️⃣1️⃣ Evaluate Model
# -------------------------
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
r2_train = metrics.r2_score(y_train, pred_train)
r2_test = metrics.r2_score(y_test, pred_test)
mae = metrics.mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred_test))
print("R² Train:", round(r2_train, 4))
print("R² Test:", round(r2_test, 4))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

# -------------------------
# 1️⃣1️⃣-a Convert Regression to Classification
# -------------------------
def classify_health(score):
    if score >= 7:
        return "Healthy"
    elif score >= 4:
        return "Moderate"
    else:
        return "Unhealthy"

y_train_class = y_train.apply(classify_health)
y_test_class = y_test.apply(classify_health)
pred_train_class = pd.Series(pred_train).apply(classify_health)
pred_test_class = pd.Series(pred_test).apply(classify_health)

# -------------------------
# 1️⃣1️⃣-b Compute Accuracy
# -------------------------

train_accuracy = accuracy_score(y_train_class, pred_train_class)
test_accuracy = accuracy_score(y_test_class, pred_test_class)
print("Classification Accuracy (Train):", round(train_accuracy, 4))
print("Classification Accuracy (Test):", round(test_accuracy, 4))

# -------------------------
# 1️⃣2️⃣ Feature Importance
# -------------------------
feat_imp = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False).head(10)
print("Top 10 Feature Importances:\n", feat_imp)

# -------------------------
# 1️⃣3️⃣ Save Model, Scaler, Encoder
# -------------------------
with open("augmented_more_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("augmented_more_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
if categorical_features:
    with open("augmented_more_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

print("Model, Scaler, and Encoder saved successfully.")
