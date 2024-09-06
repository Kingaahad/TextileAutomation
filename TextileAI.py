# Complete Script: data_model_flask_docker.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from flask import Flask, request, jsonify

# Function to preprocess data with feature engineering
def preprocess_data(data, target_col):
    X = data.drop(columns=[target_col])
    y = data[target_col]
    # Feature Engineering: Add Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_poly_df, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, poly

# Function to train and evaluate the model with hyperparameter tuning
def train_and_evaluate(model, param_grid, X_train, X_test, y_train, y_test):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    predictions = best_model.predict(X_test)
    prob_predictions = best_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, prob_predictions)
    f1 = f1_score(y_test, predictions)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, prob_predictions)
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, prob_predictions)
    
    return accuracy, roc_auc, f1, precision, recall, fpr, tpr, best_model

# Function to create and save visualizations
def create_visualizations(model_name, y_test, prob_predictions, precision, recall, fpr, tpr):
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.crosstab(y_test, (prob_predictions > 0.5).astype(int), rownames=['Actual'], colnames=['Predicted']), annot=True, fmt='d')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f'precision_recall_curve_{model_name}.png')
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.')
    plt.title(f'ROC Curve ({model_name})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f'roc_curve_{model_name}.png')
    plt.close()

# Main automation function
def automate_workflow(data, target_col):
    X_train, X_test, y_train, y_test, scaler, poly = preprocess_data(data, target_col)
    models = {
        'LogisticRegression': (LogisticRegression(), {'C': [0.1, 1, 10]}),
        'DecisionTreeClassifier': (DecisionTreeClassifier(random_state=42), {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}),
        'RandomForestClassifier': (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 10]}),
        'GradientBoostingClassifier': (GradientBoostingClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
        'SVC': (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})
    }
    results = {}
    for model_name, (model, param_grid) in models.items():
        try:
            accuracy, roc_auc, f1, precision, recall, fpr, tpr, best_model = train_and_evaluate(model, param_grid, X_train, X_test, y_train, y_test)
            results[model_name] = {'Accuracy': accuracy, 'ROC AUC': roc_auc, 'F1 Score': f1}
            print(f'{model_name}:')
            print(f'  Accuracy: {accuracy}')
            print(f'  ROC AUC: {roc_auc}')
            print(f'  F1 Score: {f1}\n')
            create_visualizations(model_name, y_test, best_model.predict_proba(X_test)[:, 1], precision, recall, fpr, tpr)
            
            # Save the best model and preprocessing tools
            joblib.dump(best_model, 'best_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            joblib.dump(poly, 'poly.pkl')
            
        except Exception as e:
            print(f'Error with {model_name}: {e}')
    return results

# Generate Larger Sample Data for Textile Production
np.random.seed(42)
textile_data_large = pd.DataFrame({
    'Machine_Age': np.random.randint(1, 10, 10000),
    'Operational_Hours': np.random.randint(1000, 5000, 10000),
    'Maintenance_Frequency': np.random.randint(1, 5, 10000),
    'Production_Volume': np.random.randint(5000, 20000, 10000),
    'Material_Quality': np.random.uniform(0.5, 1.0, 10000),
    'Employee_Shifts': np.random.randint(2, 6, 10000),
    'Order_Demand': np.random.randint(500, 4000, 10000),
    'Downtime': np.random.randint(0, 2, 10000)  # Binary outcome: 0 = No Downtime, 1 = Downtime
})

# Generate Larger Sample Data for Supply Chain Optimization
supply_chain_data_large = pd.DataFrame({
    'Supplier_Delivery_Time': np.random.randint(1, 15, 10000),
    'Inventory_Level': np.random.randint(50, 200, 10000),
    'Order_Frequency': np.random.randint(5, 30, 10000),
    'Production_Volume': np.random.randint(5000, 15000, 10000),
    'Demand_Forecast': np.random.randint(5000, 15000, 10000),
    'Supply_Chain_Issues': np.random.randint(0, 2, 10000)  # Binary outcome: 0 = No Issues, 1 = Issues
})

# Run the automated workflow for Textile Production
print("Textile Production Analysis:")
textile_results = automate_workflow(textile_data_large, 'Downtime')

# Run the automated workflow for Supply Chain Optimization
print("\nSupply Chain Optimization Analysis:")
supply_chain_results = automate_workflow(supply_chain_data_large, 'Supply_Chain_Issues')

# Flask Application for Deployment

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    df_poly = poly.transform(df)
    df_scaled = scaler.transform(df_poly)
    return df_scaled

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
