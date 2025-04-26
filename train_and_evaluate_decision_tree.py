import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Changed to DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tabulate import tabulate  # Import tabulate for tabular display

# Step 1: Define column names (if missing in CSV)
column_names = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase",
    "Total_Proteins", "Albumin", "Albumin_and_Globulin_Ratio", "Dataset"
]

# Step 2: Load the dataset
dataset = pd.read_csv("data/liver_disease.csv", names=column_names, header=None)

# Step 3: Clean column names (remove spaces)
dataset.columns = dataset.columns.str.strip()

# Step 4: Print dataset preview and check for missing values
print("🔍 First few rows of the dataset:")
print(tabulate(dataset.head(), headers='keys', tablefmt='grid'))

print("\n❌ Checking for missing values in dataset:")
missing_values = dataset.isnull().sum()
print(tabulate(missing_values.reset_index(), headers=['Column', 'Missing Values'], tablefmt='grid'))

# Step 5: Encode categorical columns
# Convert 'Gender' column to numerical values (1 = Male, 0 = Female)
le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'].astype(str))

# Step 6: Handle Missing Data (Imputation)
# Impute missing values using the mean (for numerical columns)
imputer = SimpleImputer(strategy='mean')
dataset_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)

# Step 7: Feature Scaling (Standardization)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(dataset_imputed.drop('Dataset', axis=1))

# X_scaled = dataset.drop("Dataset", axis=1)
# Step 8: Define features (X) and target (y)
X = pd.DataFrame(dataset_imputed.drop("Dataset", axis=1), columns=dataset_imputed.drop('Dataset', axis=1).columns)  # Keep column names
y = dataset_imputed['Dataset']  # Target variable (liver disease or not)

# Step 9: Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train the model (Decision Tree Classifier)
model = DecisionTreeClassifier(random_state=42)  # Use Decision Tree Classifier
model.fit(X_train, y_train)

# Step 11: Save the trained model using pickle
with open("models/decision_tree_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Display model training success
model_status = [["Model", "DecisionTreeClassifier"],
                ["Training Status", "Completed"],
                ["Model File", "models/decision_tree_model.pkl"]]

print("\n✅ Model Training Status:")
print(tabulate(model_status, headers=['Attribute', 'Value'], tablefmt='grid'))

# Check the distribution of the target class in the original dataset
class_distribution_before = dataset['Dataset'].value_counts()
print(f"\nClass distribution before preprocessing:\n{class_distribution_before}")

# Check the total number of records in the dataset
print(f"\nTotal number of records in the dataset: {len(dataset)}")
# Check the number of records in training and test sets
print(f"\nTraining set size: {len(X_train)}")
print(f"\nTest set size: {len(X_test)}")



# Step 12: Evaluate the model

# Handle NaN values in X_test (fill with X_train's mean, if necessary)
X_test = pd.DataFrame(X_test, columns=X.columns)  # Ensure X_test has column names as X
X_test.fillna(X_train.mean(), inplace=True)  # Ensures no NaN values in X_test before making predictions

# **If there are still NaN values in 'Gender' column, fill them manually**
if X_test['Gender'].isnull().sum() > 0:
    # Safely get the mode of 'Gender' in X_train, if available
    gender_mode = X_train['Gender'].mode()
    if not gender_mode.empty:
        gender_mode_value = gender_mode.iloc[0]
    else:
        gender_mode_value = 1  # Default to Male (1) if no mode is found
    X_test['Gender'].fillna(gender_mode_value, inplace=True)

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred_int = y_pred.astype(int)
# print(tabulate(pd.DataFrame(X_test[:5], columns=column_names)), y_pred_int[:5], y_test[:5])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_int)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# **Classification Report in Tabular Format**
classification_report_df = pd.DataFrame(
    classification_report(y_test, y_pred_int, zero_division=1, output_dict=True)
).transpose()
classification_report_df.index = classification_report_df.index.map(lambda x: int(float(x)) if x.replace('.', '', 1).isdigit() else x)
print("\nClassification Report :")
print(tabulate(classification_report_df, headers='keys', tablefmt='grid'))

# **Confusion Matrix in Tabular Format**
conf_matrix = confusion_matrix(y_test, y_pred_int)
conf_matrix_df = pd.DataFrame(
    conf_matrix, 
    columns=[f'Pred_{int(label)}' for label in y_test.unique()], 
    index=[f'True_{int(label)}' for label in y_test.unique()]
)

print("\nConfusion Matrix :")
print(tabulate(conf_matrix_df, headers='keys', tablefmt='grid'))

# Check for missing predicted classes
print("\nPredicted class distribution:")
print(pd.Series(y_pred_int).value_counts())

# Print out the classes where precision and recall are ill-defined
print("\nClasses with no predicted samples:")
classes_with_no_predictions = [label for label in y_test.unique() if label not in y_pred_int]
print(classes_with_no_predictions)

# -- Visualizations --

# 1. **Confusion Matrix Heatmap**
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[int(label) for label in y_test.unique()], 
            yticklabels=[int(label) for label in y_test.unique()])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 2. **Accuracy Bar Chart**
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy * 100], color='green')
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.show()

# 3. **Classification Report Heatmap**
classification_report_df.index = classification_report_df.index.map(lambda x: int(float(x)) if str(x).replace('.', '', 1).isdigit() else x)
plt.figure(figsize=(10, 6))
sns.heatmap(classification_report_df.iloc[:, :-1].astype(float), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Classification Report Heatmap')
plt.show()

# 4. **Class Distribution Bar Plot**
plt.figure(figsize=(6, 4))
sns.countplot(x=pd.Series(y_pred_int).astype(int), palette='Set2')
plt.title('Distribution of Predicted Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
