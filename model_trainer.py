import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

def train_and_save_model():
    # Load dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
    df['MedHouseVal'] = df['MedHouseVal'] * 100000  # convert to dollars

    # Simulate Ocean Proximity based on lat/lon (for demo)
    np.random.seed(42)
    conditions = [
        (df['Latitude'] > 37.5) & (df['Longitude'] < -122),
        (df['Latitude'] > 34) & (df['Latitude'] < 37) & (df['Longitude'] < -119),
        (df['Latitude'] < 34) & (df['Longitude'] > -118),
        (df['Latitude'] > 38) & (df['Longitude'] < -121),
        (df['Latitude'] > 33) & (df['Latitude'] < 34) & (df['Longitude'] > -118) & (df['Longitude'] < -117)
    ]
    choices = ['NEAR BAY', '<1H OCEAN', 'NEAR OCEAN', 'INLAND', 'ISLAND']
    df['OceanProximity'] = np.select(conditions, choices, default='INLAND')

    # Feature engineering
    df['rooms_per_hh'] = df['AveRooms'] / df['AveOccup']
    df['beds_per_room'] = df['AveBedrms'] / df['AveRooms']
    df['pop_per_hh'] = df['Population'] / df['AveOccup']
    df['income_per_hh'] = df['MedInc'] / df['AveOccup']
    df['coastal_flag'] = df['OceanProximity'].apply(lambda x: 1 if x != 'INLAND' else 0)

    # Feature columns
    feature_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                    'OceanProximity', 'rooms_per_hh', 'beds_per_room', 'pop_per_hh',
                    'income_per_hh', 'coastal_flag', 'Latitude', 'Longitude']

    X = df[feature_cols]
    y = df['MedHouseVal']

    # Preprocessing
    categorical_features = ['OceanProximity']
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

    # Pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42))
    ])

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model R² Score: {r2:.4f}")
    print(f"MAE: ${mae:.2f}")

    # Save model and metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'r2_score': r2,
        'mae': mae
    }, 'models/housing_model.pkl')

    print("Model saved to models/housing_model.pkl")
    return model, feature_cols

if __name__ == '__main__':
    train_and_save_model()