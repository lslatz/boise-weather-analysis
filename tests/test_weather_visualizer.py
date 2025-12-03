"""
Tests for the weather visualizer module.
"""

import unittest
import pandas as pd
import os
import sys
import tempfile
import shutil
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from weather_visualizer import WeatherVisualizer
from weather_analyzer import WeatherAnalyzer


class TestWeatherVisualizer(unittest.TestCase):
    """Test cases for WeatherVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.visualizer = WeatherVisualizer(output_dir=self.test_dir)
        
        # Create sample winter data
        self.winter_df = pd.DataFrame({
            'winter_year': range(2015, 2025),
            'winter_label': [f'{y-1}-{y}' for y in range(2015, 2025)],
            'avg_temp': [32.0, 33.5, 31.2, 34.0, 32.8, 33.1, 34.5, 32.5, 33.0, 34.2],
            'avg_max_temp': [40.0, 41.5, 39.2, 42.0, 40.8, 41.1, 42.5, 40.5, 41.0, 42.2],
            'avg_min_temp': [24.0, 25.5, 23.2, 26.0, 24.8, 25.1, 26.5, 24.5, 25.0, 26.2],
            'total_snowfall': [15.0, 12.5, 20.0, 10.5, 18.0, 14.0, 8.5, 16.0, 13.5, 11.0],
            'severity_score': [25.0, 22.0, 30.0, 18.0, 28.0, 24.0, 16.0, 26.0, 23.0, 20.0],
            'severity_category': ['Moderate', 'Moderate', 'Severe', 'Moderate', 'Moderate',
                                 'Moderate', 'Mild', 'Moderate', 'Moderate', 'Moderate']
        })
        
        # Create sample prediction
        self.prediction = {
            'winter_year': 2026,
            'winter_label': '2025-2026',
            'predicted_category': 'Moderate',
            'category_probabilities': {
                'Mild': 0.15,
                'Moderate': 0.65,
                'Severe': 0.20
            },
            'predicted_avg_temp': 33.0,
            'predicted_snowfall': 14.5,
            'severity_score': 24.0,
            'confidence': 0.65,
            'based_on_year': 2025,
            'input_features': {
                'summer': {
                    'temp_mean': 75.0,
                    'precip_total': 2.5,
                    'hot_days': 25
                },
                'fall': {
                    'temp_mean': 50.0,
                    'precip_total': 3.0
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove the temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test that the visualizer initializes correctly."""
        self.assertIsNotNone(self.visualizer)
        self.assertTrue(os.path.exists(self.test_dir))
    
    def test_visualize_prediction(self):
        """Test prediction visualization creation."""
        filepath = self.visualizer.visualize_prediction(self.prediction, self.winter_df)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(filepath))
        
        # Check that the file is not empty
        self.assertGreater(os.path.getsize(filepath), 0)
        
        # Check that the filename is correct
        self.assertTrue('2025-2026' in filepath)
        self.assertTrue(filepath.endswith('.png'))
    
    def test_visualize_historical_winters(self):
        """Test historical winters visualization creation."""
        filepath = self.visualizer.visualize_historical_winters(self.winter_df, n_years=10)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(filepath))
        
        # Check that the file is not empty
        self.assertGreater(os.path.getsize(filepath), 0)
        
        # Check that the filename is correct
        self.assertTrue('historical_winters' in filepath)
        self.assertTrue(filepath.endswith('.png'))
    
    def test_visualize_both(self):
        """Test creating both visualizations at once."""
        prediction_path, historical_path = self.visualizer.visualize_both(
            self.prediction, self.winter_df, n_years=10
        )
        
        # Check that both files were created
        self.assertTrue(os.path.exists(prediction_path))
        self.assertTrue(os.path.exists(historical_path))
        
        # Check that both files are not empty
        self.assertGreater(os.path.getsize(prediction_path), 0)
        self.assertGreater(os.path.getsize(historical_path), 0)
    
    def test_prediction_without_historical(self):
        """Test prediction visualization without historical data."""
        filepath = self.visualizer.visualize_prediction(self.prediction)
        
        # Should still create a file
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)


if __name__ == '__main__':
    unittest.main()
