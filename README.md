# week-2-software


Week 2 Assignment: AI for Sustainable Development
Theme: "Machine Learning Meets the UN Sustainable Development Goals (SDGs)" 🌍🤖

Objective

Design a machine learning model or AI-driven solution that addresses one of the UN SDGs using concepts from Week 2 (e.g., supervised/unsupervised learning, neural networks, NLP, or reinforcement learning). Your project should demonstrate how AI can contribute to solving global challenges like poverty, climate change, or inequality.

Need Help?

Use Week 2 materials (quiz answers, slides) for ML concepts.

Post questions in the forum with hashtag #SDGAssignment.

Check out Google Colab for cloud-based coding.

Submission

Share your GitHub repo (include Intro Readme file with screenshots  of the project demo and .py files) to finish the assignment.

Write an article of your project explaining the SDG problem you are solving and how your project brings a fitting solution to the problem. Share this article on the PLP Academy Community on the LMS.

Create a compelling elevation pitch deck for your project and share on the Groups for Peer Reviews. 

Inspiration:

“AI can be the bridge between innovation and sustainability.” — UN Tech Envoy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- 1. Simulate Data ---
def generate_simulated_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'farm_id': np.random.randint(1, 20, num_samples),
        'month': np.random.randint(1, 13, num_samples),
        'ndvi': np.random.uniform(0.2, 0.9, num_samples), # Normalized Difference Vegetation Index
        'rainfall_mm': np.random.uniform(50, 300, num_samples),
        'temperature_c': np.random.uniform(18, 32, num_samples),
        'soil_nutrient_level': np.random.uniform(0.3, 0.8, num_samples),
        'historic_yield_tons_per_ha': np.random.uniform(1.5, 5.0, num_samples)
    }
    df = pd.DataFrame(data)

    # Introduce some noise and relationship for 'actual_yield'
    df['actual_yield_tons_per_ha'] = (
        0.5 * df['ndvi'] +
        0.2 * df['rainfall_mm'] / 100 +
        0.1 * df['temperature_c'] / 10 +
        0.3 * df['soil_nutrient_level'] +
        0.5 * df['historic_yield_tons_per_ha'] +
        np.random.normal(0, 0.5, num_samples) # Add some random noise
    )
    df['actual_yield_tons_per_ha'] = np.clip(df['actual_yield_tons_per_ha'], 1.0, 7.0) # Clip to realistic range

    # Simulate some anomalies (e.g., low NDVI despite good conditions)
    num_anomalies = int(num_samples * 0.05)
    anomaly_indices = np.random.choice(df.index, num_anomalies, replace=False)
    df.loc[anomaly_indices, 'ndvi'] = np.random.uniform(0.05, 0.3, num_anomalies) # Unusually low NDVI
    df.loc[anomaly_indices, 'actual_yield_tons_per_ha'] = np.clip(df.loc[anomaly_indices, 'actual_yield_tons_per_ha'] * 0.5, 0.5, 3.0) # Significantly reduced yield

    return df

# --- 2. Crop Yield Prediction (Supervised Learning - Regression) ---
def train_yield_prediction_model(df):
    features = ['ndvi', 'rainfall_mm', 'temperature_c', 'soil_nutrient_level', 'historic_yield_tons_per_ha']
    target = 'actual_yield_tons_per_ha'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Yield Prediction Model MSE: {mse:.2f}")

    return model, features, X_test, y_test, predictions

# --- 3. Anomaly Detection (Unsupervised Learning) ---
def train_anomaly_detection_model(df):
    # For anomaly detection, we often look at features that indicate health or deviation
    # Using 'ndvi' and 'actual_yield_tons_per_ha' to detect unusual patterns
    anomaly_features = ['ndvi', 'actual_yield_tons_per_ha', 'rainfall_mm']

    # Train Isolation Forest on the 'normal' data distribution
    # We assume most data is normal, and Isolation Forest finds outliers
    model = IsolationForest(contamination=0.05, random_state=42) # contamination is the proportion of outliers in the dataset
    model.fit(df[anomaly_features])

    # Predict anomalies (-1 for outliers, 1 for inliers)
    df['anomaly_score'] = model.decision_function(df[anomaly_features])
    df['is_anomaly'] = model.predict(df[anomaly_features]) # -1 for outlier, 1 for inlier

    return model, df

# --- Demo Run ---
if __name__ == "__main__":
    print("--- Generating Simulated Crop Data ---")
    simulated_data = generate_simulated_data(num_samples=1500)
    print(simulated_data.head())
    print(f"Total anomalies simulated: {simulated_data[simulated_data['ndvi'] < 0.3].shape[0]}") # Count based on our simulation rule

    print("\n--- Training Crop Yield Prediction Model ---")
    yield_model, features, X_test, y_test, predictions = train_yield_prediction_model(simulated_data)

    # Demonstrate a prediction for a new, hypothetical data point
    print("\n--- Demonstrating a new yield prediction ---")
    new_data_point = pd.DataFrame([[0.75, 150, 25, 0.6, 3.0]], columns=features)
    predicted_yield = yield_model.predict(new_data_point)[0]
    print(f"Predicted yield for hypothetical farm: {predicted_yield:.2f} tons/ha")

    # Visualize actual vs predicted yields
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Yield")
    plt.ylabel("Predicted Yield")
    plt.title("Actual vs. Predicted Crop Yields")
    plt.grid(True)
    plt.show()


    print("\n--- Training Anomaly Detection Model ---")
    anomaly_model, df_with_anomalies = train_anomaly_detection_model(simulated_data.copy()) # Use a copy to not modify original

    print("\n--- Sample Anomalies Detected ---")
    anomalies_detected = df_with_anomalies[df_with_anomalies['is_anomaly'] == -1]
    print(f"Number of anomalies detected: {anomalies_detected.shape[0]}")
    print("Example detected anomalies:")
    print(anomalies_detected[['farm_id', 'month', 'ndvi', 'rainfall_mm', 'actual_yield_tons_per_ha', 'is_anomaly']].head())

    # Visualize anomalies (e.g., NDVI vs Yield with anomaly points highlighted)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_with_anomalies['ndvi'], df_with_anomalies['actual_yield_tons_per_ha'],
                c=df_with_anomalies['is_anomaly'], cmap='coolwarm', alpha=0.7)
    plt.xlabel("NDVI")
    plt.ylabel("Actual Yield (tons/ha)")
    plt.title("Anomaly Detection: NDVI vs. Yield (-1 = Anomaly)")
    plt.colorbar(label='Anomaly Status')
    plt.grid(True)
    plt.show()

    print("\n--- Project Demo Complete ---")
    print("This simple demonstration shows how AI can predict yields and flag potential issues to aid in food security.")
