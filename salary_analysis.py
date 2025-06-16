import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("glassdoor_jobs.csv")

# Drop unnecessary column
if 'Unnamed: 0' in df.columns:
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

# Remove rows with invalid salary
df = df[df['Salary Estimate'] != '-1']

# Parse salary estimates
def parse_salary(sal):
    sal = sal.lower()
    hourly = 1 if 'per hour' in sal else 0
    employer_provided = 1 if 'employer provided salary:' in sal else 0
    sal = sal.replace('(glassdoor est.)', '').replace('employer provided salary:', '')
    sal = sal.replace('$', '').replace('k', '').replace('per hour', '').replace(',', '')
    try:
        min_sal, max_sal = sal.split('-')
        return float(min_sal.strip()), float(max_sal.strip()), hourly, employer_provided
    except:
        return np.nan, np.nan, hourly, employer_provided

df[['min_salary', 'max_salary', 'hourly', 'employer_provided']] = df['Salary Estimate'].apply(
    lambda x: pd.Series(parse_salary(x))
)

# Drop rows with parsing errors
df.dropna(subset=['min_salary', 'max_salary'], inplace=True)

# Calculate average salary
df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

# Extract state
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[-1].strip())

# Simplify job title
def clean_title(title):
    title = title.lower()
    if 'data scientist' in title:
        return 'data scientist'
    elif 'data engineer' in title:
        return 'data engineer'
    elif 'analyst' in title:
        return 'analyst'
    elif 'machine learning' in title:
        return 'ml engineer'
    elif 'manager' in title:
        return 'manager'
    elif 'director' in title:
        return 'director'
    elif 'software engineer' in title or 'developer' in title:
        return 'software engineer'
    else:
        return 'other'

df['job_title_simplified'] = df['Job Title'].apply(clean_title)

# Extract seniority
def seniority(title):
    title = title.lower()
    if 'senior' in title or 'sr' in title:
        return 'senior'
    elif 'junior' in title or 'jr' in title:
        return 'junior'
    elif 'lead' in title or 'principal' in title:
        return 'lead'
    else:
        return 'na'

df['seniority'] = df['Job Title'].apply(seniority)

# Select relevant features
features = [
    'avg_salary', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector',
    'Revenue', 'job_state', 'hourly', 'employer_provided', 'job_title_simplified', 'seniority'
]
df_model = df[features]

# One-hot encoding
df_dum = pd.get_dummies(df_model, drop_first=True)

# Split into X, y
X = df_dum.drop('avg_salary', axis=1)
y = df_dum['avg_salary']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search parameters
param_grid = {
    'n_estimators': [200],  # Based on best params
    'max_depth': [None],
    'min_samples_split': [2]
}

# GridSearchCV
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

print("✅ Best Parameters:", grid_rf.best_params_)

# Train and test predictions
train_preds = best_rf.predict(X_train)
test_preds = best_rf.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

print("\n--- 📊 Model Evaluation ---")
print(f"Train R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# Plot Overfitting/Underfitting
plt.figure(figsize=(6, 4))
plt.bar(['Train R²', 'Test R²'], [train_r2, test_r2], color=['green', 'orange'])
plt.title("Overfitting/Underfitting Check (R² Score)")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()

# Cross-validation R² scores
cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='r2')
print("\n--- 🔁 Cross-Validation R² Scores ---")
print(f"R² Scores: {cv_scores}")
print(f"Average CV R²: {np.mean(cv_scores):.4f}")

# ----------- ADDITIONAL VISUALIZATIONS -----------

# 1. Actual vs Predicted Scatter Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, test_preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Residuals Plot
residuals = y_test - test_preds
plt.figure(figsize=(6,4))
plt.scatter(test_preds, residuals, alpha=0.5)
plt.hlines(0, xmin=test_preds.min(), xmax=test_preds.max(), colors='r', linestyles='dashed')
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Feature Importance Plot
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# 4. Distribution of Residuals
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.tight_layout()
plt.show()

# 5. Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    best_rf, X, y, cv=5, scoring='r2', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5), random_state=42)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(6,4))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
