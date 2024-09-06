🛠️ Machine Learning Automation with Flask & Docker

🚀 Overview
This project automates machine learning model training, evaluation, and deployment using Flask and Docker. It allows you to:

- Preprocess data, including adding polynomial features and scaling them.
- Train and tune various machine learning models using GridSearchCV.
- Generate evaluation metrics like Accuracy, ROC AUC, F1 Score, and visualizations such as Confusion Matrix, Precision-Recall Curve, and ROC Curve.
- Save the best model and preprocessing steps for future predictions.
- Serve the model using a Flask API to predict new data points.

📂 Project Structure
```
data_model_flask_docker.py   # Main script containing all logic (ML & Flask)
requirements.txt             # Python dependencies
Dockerfile                   # Docker configuration to containerize the Flask app
README.md                    # This README file
```

🧠 Key Features

1. Preprocessing 🔧
The `preprocess_data()` function handles:

- Splitting data into features (`X`) and target (`y`).
- Adding **polynomial features** to capture more complex relationships in the data.
- **Scaling** features using `StandardScaler` to normalize them.

### 2. **Training & Evaluation** 📊
The `train_and_evaluate()` function trains various models with hyperparameter tuning using `GridSearchCV`. Supported models include:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Classifier (SVC)

Evaluation metrics:
- Accuracy
- ROC AUC
- F1 Score
- Precision-Recall Curve & ROC Curve visualizations

3. Visualization 📈
The `create_visualizations()` function generates visualizations to evaluate model performance:
- Confusion Matrix
- Precision-Recall Curve
- ROC Curve

These graphs are saved as PNG files for further analysis.

4. Deployment with Flask 🌐
The Flask app hosts a `/predict` API endpoint. You can send **JSON POST** requests to predict outcomes using new data points. The model and preprocessing tools (scaler, polynomial transformer) are loaded from saved `.pkl` files.

📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```

2. Install dependencies:
Ensure you have Python 3.7+ and pip installed. Then, run:
```bash
pip install -r requirements.txt
```

3. Run the Flask app:
Start the Flask server locally:
```bash
python data_model_flask_docker.py
```

Access it at `http://127.0.0.1:5000/`.

4. Build and run with Docker:
Alternatively, you can use Docker for containerization:
```bash
docker build -t ml_flask_app .
docker run -p 5000:5000 ml_flask_app
```

🔄 API Usage

Endpoint: `/predict`

Send a POST request with a JSON payload containing feature values. For example:

```json
{
  "Machine_Age": 5,
  "Operational_Hours": 3500,
  "Maintenance_Frequency": 3,
  "Production_Volume": 12000,
  "Material_Quality": 0.85,
  "Employee_Shifts": 4,
  "Order_Demand": 2500
}
```

Response:
```json
{
  "prediction": 1
}
```
The response is a binary prediction (e.g., `1` for Downtime, `0` for No Downtime).

⚙️ Models Trained
Here’s a summary of models trained and their hyperparameters:

- Logistic Regression: `C = [0.1, 1, 10]`
- Decision Tree Classifier: `max_depth = [3, 5, 7, 10]`, `min_samples_split = [2, 5, 10]`
- Random Forest Classifier: `n_estimators = [50, 100, 200]`, `max_depth = [3, 5, 7, 10]`
- Gradient Boosting Classifier: `n_estimators = [50, 100, 200]`, `learning_rate = [0.01, 0.1, 0.2]`
- Support Vector Classifier (SVC): `C = [0.1, 1, 10]`, `kernel = ['linear', 'rbf']`

📊 Example Use Cases

Textile Production Downtime Prediction:
- Predict if machine downtime will occur based on various features such as Machine Age, Operational Hours, and Material Quality.

Supply Chain Optimization:
- Predict whether supply chain issues will occur based on features like Supplier Delivery Time, Inventory Level, and Order Frequency.

📝 License
This project is licensed under the MIT License. Feel free to contribute, fork, or use it for your own projects!

