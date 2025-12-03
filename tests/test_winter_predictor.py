"""
Tests for the winter predictor module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from winter_predictor import WinterPredictor


class TestWinterPredictor(unittest.TestCase):
    """Test cases for WinterPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = WinterPredictor()
        
        # Create sample correlation data for training
        n_samples = 30
        
        # Generate mutually exclusive ENSO phases
        enso_phases = np.random.choice([0, 1, 2], n_samples)  # 0=El Niño, 1=La Niña, 2=Neutral
        enso_el_nino = (enso_phases == 0).astype(float)
        enso_la_nina = (enso_phases == 1).astype(float)
        enso_neutral = (enso_phases == 2).astype(float)
        
        self.sample_correlation_df = pd.DataFrame({
            "winter_year": range(1990, 1990 + n_samples),
            "prev_summer_temp": np.random.uniform(70, 90, n_samples),
            "prev_summer_precip": np.random.uniform(1, 5, n_samples),
            "prev_summer_hot_days": np.random.randint(0, 30, n_samples),
            "prev_fall_temp": np.random.uniform(40, 60, n_samples),
            "prev_fall_precip": np.random.uniform(2, 6, n_samples),
            "enso_oni": np.random.uniform(-2, 2, n_samples),
            "enso_el_nino": enso_el_nino,
            "enso_la_nina": enso_la_nina,
            "enso_neutral": enso_neutral,
            "winter_severity": np.random.uniform(10, 50, n_samples),
            "winter_snowfall": np.random.uniform(5, 30, n_samples),
            "winter_temp_avg": np.random.uniform(25, 40, n_samples),
            "winter_category": np.random.choice(["Mild", "Moderate", "Severe"], n_samples)
        })
    
    def test_initialization(self):
        """Test that the predictor initializes correctly."""
        self.assertIsNone(self.predictor.severity_model)
        self.assertIsNone(self.predictor.snowfall_model)
        self.assertIsNone(self.predictor.temperature_model)
        self.assertFalse(self.predictor.is_trained)
    
    def test_prepare_training_data(self):
        """Test training data preparation."""
        X, y_severity, y_snowfall, y_temp, y_category = self.predictor.prepare_training_data(
            self.sample_correlation_df
        )
        
        # Check shapes
        self.assertEqual(len(X), len(y_severity))
        self.assertEqual(len(X), len(y_snowfall))
        self.assertEqual(len(X), len(y_temp))
        self.assertEqual(len(X), len(y_category))
        
        # Check that feature columns are set
        self.assertTrue(len(self.predictor.feature_columns) > 0)
    
    def test_train(self):
        """Test model training."""
        self.predictor.train(self.sample_correlation_df)
        
        # Check that models are trained
        self.assertIsNotNone(self.predictor.severity_model)
        self.assertIsNotNone(self.predictor.snowfall_model)
        self.assertIsNotNone(self.predictor.temperature_model)
        self.assertIsNotNone(self.predictor.category_model)
        self.assertTrue(self.predictor.is_trained)
    
    def test_predict(self):
        """Test making predictions."""
        # Train first
        self.predictor.train(self.sample_correlation_df)
        
        # Make prediction
        summer_features = {
            "temp_mean": 75.0,
            "precip_total": 3.0,
            "hot_days": 15
        }
        fall_features = {
            "temp_mean": 50.0,
            "precip_total": 4.0
        }
        enso_features = {
            "oni": 1.5,
            "el_nino": 1.0,
            "la_nina": 0.0,
            "neutral": 0.0
        }
        
        prediction = self.predictor.predict(summer_features, fall_features, enso_features)
        
        # Check that prediction has expected keys
        expected_keys = ["severity_score", "predicted_category", "category_probabilities",
                        "predicted_snowfall", "predicted_avg_temp", "confidence"]
        for key in expected_keys:
            self.assertTrue(key in prediction, f"Missing key: {key}")
        
        # Check that values are reasonable
        self.assertIsInstance(prediction["severity_score"], float)
        self.assertIsInstance(prediction["predicted_category"], str)
        self.assertIsInstance(prediction["category_probabilities"], dict)
        self.assertGreater(prediction["confidence"], 0)
        self.assertLessEqual(prediction["confidence"], 1)
    
    def test_predict_without_training(self):
        """Test that prediction fails without training."""
        summer_features = {"temp_mean": 75.0}
        
        with self.assertRaises(ValueError):
            self.predictor.predict(summer_features)
    
    def test_enso_features_included(self):
        """Test that ENSO features are included in the model."""
        self.predictor.train(self.sample_correlation_df)
        
        # Check that ENSO features are in the feature columns
        self.assertIn("enso_oni", self.predictor.feature_columns)
        self.assertIn("enso_el_nino", self.predictor.feature_columns)
        self.assertIn("enso_la_nina", self.predictor.feature_columns)
        self.assertIn("enso_neutral", self.predictor.feature_columns)
        
        # Check that all 9 features are included (5 weather + 4 ENSO)
        self.assertEqual(len(self.predictor.feature_columns), 9)
    
    def test_feature_importance(self):
        """Test that feature importance can be retrieved."""
        self.predictor.train(self.sample_correlation_df)
        
        # Check that we can get feature importances
        importances = self.predictor.severity_model.feature_importances_
        self.assertEqual(len(importances), len(self.predictor.feature_columns))
        
        # All importances should sum to approximately 1
        self.assertAlmostEqual(sum(importances), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
