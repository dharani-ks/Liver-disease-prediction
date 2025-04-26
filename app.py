from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

app = Flask(__name__)

# Load trained models
with open('models/decision_tree_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('models/random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

with open('models/svm_model.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

def determine_liver_disease_type(total_bilirubin, direct_bilirubin, alkaline_phosphotase, 
                           alamine_aminotransferase, aspartate_aminotransferase, 
                           total_proteins, albumin, albumin_and_globulin_ratio):
    
    # Check for Hepatitis (Acute or Chronic)
    
    # Healthy liver check - this is a new check for healthy liver based on normal ranges
    if (total_bilirubin <= 1.2 and direct_bilirubin <= 0.3 and alkaline_phosphotase <= 147 and 
        alamine_aminotransferase <= 56 and aspartate_aminotransferase <= 40 and 
        total_proteins >= 6.0 and albumin >= 3.5 and albumin_and_globulin_ratio >= 1.0):
        return "No liver disease"  # Return this if all values are within normal ranges.
    
    # Hepatitis (Acute or Chronic)
    if total_bilirubin > 1.2 and direct_bilirubin > 0.5 and alkaline_phosphotase > 200 and \
       alamine_aminotransferase > 1000 and aspartate_aminotransferase > 1000 and \
       albumin < 3.5 and albumin_and_globulin_ratio < 1.0:
        return "Hepatitis"
    
    # Cholestatic Liver Disease
    elif total_bilirubin > 1.2 and direct_bilirubin > 0.5 and alkaline_phosphotase > 200:
        return "Cholestatic Liver Disease"
    
    # Cirrhosis
    elif total_bilirubin > 1.5 and direct_bilirubin > 0.5 and alkaline_phosphotase > 200 and \
         albumin < 3.5 and albumin_and_globulin_ratio < 1.0 and aspartate_aminotransferase > 200:
        return "Cirrhosis"
    
    # General liver dysfunction (if none of the above match, but there are mild abnormalities)
    else:
        return "General liver dysfunction"


# Function to provide advice
def get_advice(disease_type):
    advice = {
        "No liver disease": "Your liver is healthy. Continue to maintain a healthy lifestyle with a balanced diet and regular exercise.",
        "Hepatitis": "Seek medical help immediately. Avoid alcohol, follow a prescribed treatment plan, and monitor your liver function regularly.",
        "Cholestatic Liver Disease": "Maintain a healthy diet, stay hydrated, and avoid fatty foods. Consult your doctor for further treatment options.",
        "Cirrhosis": "Stop alcohol consumption immediately, follow a liver-friendly diet, and consult with a hepatologist. Regular check-ups are crucial.",
        "General liver dysfunction": "Improve your diet, exercise regularly, and reduce the use of medications that may strain the liver. Get a medical check-up for further investigation."
    }
    return advice.get(disease_type, "Maintain a healthy lifestyle and get regular check-ups.")

#possilble causes
def get_possible_causes(disease_type):
    causes = {
        "No liver disease": ["Healthy lifestyle, balanced diet, and regular physical activity."],
        "Hepatitis": [
            "Viral infections (e.g., Hepatitis A, B, C).",
            "Excessive alcohol consumption.",
            "Medications or toxins that affect the liver.",
            "Autoimmune conditions where the immune system attacks liver cells.",
            "Unprotected sex or needle-sharing (for viral Hepatitis)."
        ],
        "Cholestatic Liver Disease": [
            "Obstruction in bile ducts (e.g., gallstones, bile duct stricture).",
            "Liver diseases such as primary biliary cirrhosis or primary sclerosing cholangitis.",
            "Excessive alcohol intake leading to liver damage.",
            "Chronic liver diseases leading to bile flow issues."
        ],
        "Cirrhosis": [
            "Chronic alcohol consumption.",
            "Chronic viral infections (e.g., Hepatitis B, C).",
            "Non-alcoholic fatty liver disease (NAFLD).",
            "Hemochromatosis (excess iron accumulation in the liver).",
            "Autoimmune liver diseases.",
            "Long-term use of certain medications or toxins."
        ],
        "General liver dysfunction": [
            "Obesity or being overweight.",
            "Excessive alcohol consumption.",
            "Chronic use of medications that strain the liver.",
            "Infections or viral liver diseases.",
            "Dietary factors like high-fat foods or high sugar intake.",
            "Metabolic conditions like fatty liver disease."
        ]
    }
    return causes.get(disease_type, ["Maintain a healthy lifestyle and get regular check-ups."])





# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        age = float(request.form['age'])
        gender = request.form['gender']
        total_bilirubin = float(request.form['total_bilirubin'])
        direct_bilirubin = float(request.form['direct_bilirubin'])
        alkaline_phosphotase = float(request.form['alkaline_phosphotase'])
        alamine_aminotransferase = float(request.form['alamine_aminotransferase'])
        aspartate_aminotransferase = float(request.form['aspartate_aminotransferase'])
        total_proteins = float(request.form['total_proteins'])
        albumin = float(request.form['albumin'])
        albumin_and_globulin_ratio = float(request.form['albumin_and_globulin_ratio'])
        model_choice = request.form['model_choice']

        # Encode Gender (0 for Female, 1 for Male)
        gender_encoded = 1 if gender.lower() == "male" else 0

        # Create input array
        input_data = np.array([[age, gender_encoded, total_bilirubin, direct_bilirubin,
                                alkaline_phosphotase, alamine_aminotransferase, aspartate_aminotransferase,
                                total_proteins, albumin, albumin_and_globulin_ratio]])

        print(input_data)
        # Initialize imputer and scaler
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        # Apply imputer and scaler
        input_data_imputed = imputer.fit_transform(input_data)
        # input_data_scaled = scaler.fit_transform(input_data_imputed)

        # Select model based on user choice
        if model_choice == "decision_tree":
            model = decision_tree_model
            accuracy = "Accuracy: 85%"
        elif model_choice == "random_forest":
            model = random_forest_model
            accuracy = "Accuracy: 90%"
        elif model_choice == "svm":
            model = svm_model
            accuracy = "Accuracy: 88%"
        else:
            return jsonify({"error": "Invalid model selection"})

        # Make a prediction
        print(model)
        print(input_data_imputed)
        prediction = model.predict(input_data_imputed)
        print(prediction)
        prediction = prediction[0]
        prediction = int(prediction) 
        print(f"Prediction value: {prediction}")
        # Prepare output based on prediction
        if prediction == 2:
            return render_template('result.html', 
                           result="The person has NO liver disease",
                           result_color="green",  # Pass color info to the template
                           advice="Maintain a healthy lifestyle and get regular check-ups.",
                           disease_type="No liver disease",
                           causes="",
                           accuracy=accuracy)
        else:
            disease_type = determine_liver_disease_type(total_bilirubin, direct_bilirubin, alkaline_phosphotase, 
                       alamine_aminotransferase, aspartate_aminotransferase, 
                       total_proteins, albumin, albumin_and_globulin_ratio)
            advice = get_advice(disease_type)
            causes = get_possible_causes(disease_type)
            return render_template('result.html', 
                           result="The person has liver disease.",
                           result_color="red",  # Pass color info to the template
                           disease_type=disease_type,
                           causes=causes,
                           advice=advice,
                           accuracy=accuracy)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
