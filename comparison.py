import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tabulate import tabulate

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

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(dataset_imputed.drop('Dataset', axis=1))

# Define features (X) and target (y)
X = pd.DataFrame(X_scaled, columns=dataset_imputed.drop('Dataset', axis=1).columns)
y = dataset_imputed['Dataset']  # Target variable (liver disease or not)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model paths
model_paths = {
    "Decision Tree": "models/decision_tree_model.pkl",  # replace with actual model path
    "Random Forest": "models/random_forest_model.pkl",  # replace with actual model path
    "SVM": "models/svm_model.pkl"  # replace with actual model path
}

# Store predictions and accuracies
results = {}

# Load models and make predictions
for model_name, model_path in model_paths.items():
    # Load the model from the .pkl file
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
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

# Identify the model with the highest accuracy
best_model = results_df['Accuracy (%)'].idxmax()    
best_accuracy = results_df.loc[best_model, 'Accuracy (%)']
print(f"\nThe best model is: {best_model} with an accuracy of {best_accuracy:.2f}%")

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
