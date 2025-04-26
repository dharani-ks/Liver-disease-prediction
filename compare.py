import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from tabulate import tabulate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load and preprocess the dataset
column_names = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase",
    "Total_Proteins", "Albumin", "Albumin_and_Globulin_Ratio", "Dataset"
]

# Load dataset
dataset = pd.read_csv("data/liver_disease.csv", names=column_names, header=None)

# Clean column names
dataset.columns = dataset.columns.str.strip()

# Encode categorical columns
le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'].astype(str))

# Handle Missing Data (Imputation)
imputer = SimpleImputer(strategy='mean')
dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

# Define features (X) and target (y)
X = dataset_imputed.drop('Dataset', axis=1)
y = dataset_imputed['Dataset']  # Target variable (liver disease or not)

# Feature Selection (Selecting top 8 features)
selector = SelectKBest(score_func=f_classif, k=8)
X_selected = selector.fit_transform(X, y)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define models with hyperparameter tuning
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# Hyperparameter tuning
param_grid = {
    "Decision Tree": {"max_depth": [5, 10, 15], "min_samples_split": [2, 5, 10]},
    "Random Forest": {"n_estimators": [100, 200, 300], "max_depth": [10, 20, None]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
}

best_models = {}
results = {}

for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    affected = sum(y_pred == 1)  # class 1 = affected
    not_affected = sum(y_pred == 2)  # class 2 = not affected
    
    results[model_name] = {
        "Affected": affected,
        "Not Affected": not_affected,
        "Accuracy (%)": accuracy * 100  # Convert to percentage
    }

# Convert the results to a DataFrame for easy viewing
results_df = pd.DataFrame(results).T

# Display the results in tabular format using tabulate
print("Model Comparison (Affected vs. Not Affected & Accuracy):")
print(tabulate(results_df, headers='keys', tablefmt='grid'))

# Identify the best model
best_model_name = results_df['Accuracy (%)'].idxmax()
best_accuracy = results_df.loc[best_model_name, 'Accuracy (%)']
print(f"\nThe best model is: {best_model_name} with an accuracy of {best_accuracy:.2f}% 🚀")

# Visualization of the results
plt.figure(figsize=(12, 6))

# Subplot 1: Accuracy Comparison Bar Plot
plt.subplot(1, 2, 1)
results_df['Accuracy (%)'].plot(kind='bar', color=['lightblue', 'lightgreen', 'lightcoral'], figsize=(8, 5))
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=0)

# Subplot 2: Affected vs Not Affected for each model
plt.subplot(1, 2, 2)
affected_counts = [results[model_name]['Affected'] for model_name in results.keys()]
not_affected_counts = [results[model_name]['Not Affected'] for model_name in results.keys()]

bar_width = 0.35
index = range(len(results))

plt.bar(index, affected_counts, bar_width, label='Affected', color='green')
plt.bar([i + bar_width for i in index], not_affected_counts, bar_width, label='Not Affected', color='red')

plt.xlabel('Model')
plt.ylabel('Count')
plt.title('Affected vs Not Affected')
plt.xticks([i + bar_width / 2 for i in index], results.keys())
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
