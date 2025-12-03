"""
Winter prediction model using historical weather patterns.
Predicts winter conditions based on preceding seasonal data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class WinterPredictor:
    """Predicts winter conditions based on historical patterns."""
    
    def __init__(self):
        """Initialize the winter predictor."""
        self.severity_model = None
        self.snowfall_model = None
        self.temperature_model = None
        self.category_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.feature_means = {}  # Store historical means for imputation
        self.is_trained = False
    
    def prepare_training_data(self, correlation_df):
        """Prepare data for model training.
        
        Args:
            correlation_df: DataFrame with seasonal correlations
            
        Returns:
            tuple: (X, y_severity, y_snowfall, y_temp, y_category)
        """
        # Select features for prediction
        feature_cols = []
        for col in correlation_df.columns:
            if col.startswith("prev_"):
                feature_cols.append(col)
        
        self.feature_columns = feature_cols
        
        # Remove rows with missing values
        df_clean = correlation_df[feature_cols + [
            "winter_severity", "winter_snowfall", "winter_temp_avg", "winter_category"
        ]].dropna()
        
        if len(df_clean) < 10:
            raise ValueError("Insufficient data for training (need at least 10 complete records)")
        
        # Calculate feature means for imputation
        self.feature_means = df_clean[feature_cols].mean().to_dict()
        
        X = df_clean[feature_cols].values
        y_severity = df_clean["winter_severity"].values
        y_snowfall = df_clean["winter_snowfall"].values
        y_temp = df_clean["winter_temp_avg"].values
        y_category = df_clean["winter_category"].values
        
        return X, y_severity, y_snowfall, y_temp, y_category
    
    def train(self, correlation_df):
        """Train prediction models on historical data.
        
        Args:
            correlation_df: DataFrame with seasonal correlations
        """
        print("Training winter prediction models...")
        
        X, y_severity, y_snowfall, y_temp, y_category = self.prepare_training_data(
            correlation_df
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train severity regression model
        self.severity_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=3
        )
        self.severity_model.fit(X_scaled, y_severity)
        
        # Train snowfall regression model
        self.snowfall_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=3
        )
        self.snowfall_model.fit(X_scaled, y_snowfall)
        
        # Train temperature regression model
        self.temperature_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=3
        )
        self.temperature_model.fit(X_scaled, y_temp)
        
        # Train category classification model
        self.category_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=3
        )
        self.category_model.fit(X_scaled, y_category)
        
        self.is_trained = True
        
        print(f"Models trained on {len(X)} historical winters")
        print(f"Features used: {', '.join(self.feature_columns)}")
        
        # Calculate and print feature importance
        self._print_feature_importance()
    
    def _print_feature_importance(self):
        """Print feature importance from the severity model."""
        importances = self.severity_model.feature_importances_
        feature_importance = sorted(
            zip(self.feature_columns, importances),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("\nTop features for winter severity prediction:")
        for i, (feature, importance) in enumerate(feature_importance[:5], 1):
            print(f"  {i}. {feature}: {importance:.3f}")
    
    def predict(self, summer_features, fall_features=None):
        """Predict winter conditions based on summer (and optionally fall) data.
        
        Args:
            summer_features: Dictionary with summer weather features
            fall_features: Optional dictionary with fall weather features
            
        Returns:
            dict: Predicted winter conditions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Build feature vector
        feature_vector = {}
        
        # Add summer features
        if "temp_mean" in summer_features:
            feature_vector["prev_summer_temp"] = summer_features["temp_mean"]
        if "precip_total" in summer_features:
            feature_vector["prev_summer_precip"] = summer_features["precip_total"]
        if "hot_days" in summer_features:
            feature_vector["prev_summer_hot_days"] = summer_features["hot_days"]
        
        # Add fall features if provided
        if fall_features:
            if "temp_mean" in fall_features:
                feature_vector["prev_fall_temp"] = fall_features["temp_mean"]
            if "precip_total" in fall_features:
                feature_vector["prev_fall_precip"] = fall_features["precip_total"]
        
        # Ensure all required features are present
        X = []
        for col in self.feature_columns:
            if col in feature_vector:
                X.append(feature_vector[col])
            else:
                # Use historical mean for missing features
                X.append(self.feature_means.get(col, 0))
        
        X = np.array([X])
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        severity_pred = self.severity_model.predict(X_scaled)[0]
        snowfall_pred = self.snowfall_model.predict(X_scaled)[0]
        temp_pred = self.temperature_model.predict(X_scaled)[0]
        category_pred = self.category_model.predict(X_scaled)[0]
        category_probs = self.category_model.predict_proba(X_scaled)[0]
        
        # Get category probabilities
        categories = self.category_model.classes_
        category_prob_dict = {cat: prob for cat, prob in zip(categories, category_probs)}
        
        return {
            "severity_score": float(severity_pred),
            "predicted_category": category_pred,
            "category_probabilities": category_prob_dict,
            "predicted_snowfall": float(snowfall_pred),
            "predicted_avg_temp": float(temp_pred),
            "confidence": float(max(category_probs))
        }
    
    def predict_from_current_data(self, analyzer, target_winter_year=None):
        """Predict winter conditions based on current year's data.
        
        Args:
            analyzer: WeatherAnalyzer instance with current data
            target_winter_year: Year of winter to predict (e.g., 2026 for 2025-2026 winter)
                              If None, predicts the upcoming winter
            
        Returns:
            dict: Predicted winter conditions with context
        """
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        if target_winter_year is None:
            # If we're past July, predict next year's winter
            # Otherwise predict this year's winter
            if current_month >= 7:
                target_winter_year = current_year + 1
            else:
                target_winter_year = current_year
        
        # Get current year's seasonal data
        seasonal_df = analyzer.calculate_seasonal_features()
        current_seasonal = seasonal_df[seasonal_df["year"] == target_winter_year - 1]
        
        if len(current_seasonal) == 0:
            # Try to use most recent complete year
            current_seasonal = seasonal_df[seasonal_df["year"] == seasonal_df["year"].max()]
        
        # Extract summer and fall features
        summer_features = {}
        fall_features = {}
        
        if len(current_seasonal) > 0:
            row = current_seasonal.iloc[0]
            
            if "summer_temp_mean" in row:
                summer_features["temp_mean"] = row["summer_temp_mean"]
                summer_features["precip_total"] = row["summer_precip_total"]
                summer_features["hot_days"] = row["summer_hot_days"]
            
            if "fall_temp_mean" in row:
                fall_features["temp_mean"] = row["fall_temp_mean"]
                fall_features["precip_total"] = row["fall_precip_total"]
        
        # Make prediction
        prediction = self.predict(summer_features, fall_features)
        
        # Add context
        prediction["winter_year"] = target_winter_year
        prediction["winter_label"] = f"{target_winter_year - 1}-{target_winter_year}"
        prediction["based_on_year"] = target_winter_year - 1
        
        # Add input features for transparency
        prediction["input_features"] = {
            "summer": summer_features,
            "fall": fall_features
        }
        
        return prediction
    
    def get_historical_comparison(self, prediction, winter_df):
        """Compare prediction to historical winters.
        
        Args:
            prediction: Dictionary with prediction results
            winter_df: DataFrame with historical winter data
            
        Returns:
            dict: Comparison statistics
        """
        # Find similar historical winters
        severity_score = prediction["severity_score"]
        
        similar_winters = winter_df[
            (winter_df["severity_score"] >= severity_score - 5) &
            (winter_df["severity_score"] <= severity_score + 5)
        ].sort_values("winter_year", ascending=False)
        
        comparison = {
            "similar_winters": similar_winters[["winter_label", "severity_category", 
                                                "avg_temp", "total_snowfall"]].head(5).to_dict("records"),
            "historical_avg_severity": float(winter_df["severity_score"].mean()),
            "historical_avg_snowfall": float(winter_df["total_snowfall"].mean()),
            "historical_avg_temp": float(winter_df["avg_temp"].mean()),
            "percentile_rank": float((winter_df["severity_score"] < severity_score).sum() / len(winter_df) * 100)
        }
        
        return comparison
