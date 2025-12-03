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
                                 'Moderate', 'Mild', 'Moderate', 'Moderate', 'Moderate'],
            'enso_phase': ['El Niño', 'El Niño', 'Neutral', 'La Niña', 'Neutral',
                          'La Niña', 'Neutral', 'El Niño', 'La Niña', 'El Niño']
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
        
        # Create sample ENSO info
        self.enso_info = {
            'phase': 'La Niña',
            'strength': 'Weak',
            'oni_value': -0.5,
            'description': 'Weak La Niña (ONI: -0.5°C)'
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
    
    def test_visualize_prediction_with_enso(self):
        """Test prediction visualization with ENSO information."""
        filepath = self.visualizer.visualize_prediction(
            self.prediction, self.winter_df, self.enso_info
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)
    
    def test_visualize_both_with_enso(self):
        """Test creating both visualizations with ENSO information."""
        prediction_path, historical_path = self.visualizer.visualize_both(
            self.prediction, self.winter_df, n_years=10, enso_info=self.enso_info
        )
        
        # Check that both files were created
        self.assertTrue(os.path.exists(prediction_path))
        self.assertTrue(os.path.exists(historical_path))
        
        # Check that both files are not empty
        self.assertGreater(os.path.getsize(prediction_path), 0)
        self.assertGreater(os.path.getsize(historical_path), 0)
    
    def test_historical_winters_with_enso_data(self):
        """Test that historical winters visualization includes ENSO data."""
        # The winter_df already has enso_phase column
        filepath = self.visualizer.visualize_historical_winters(self.winter_df, n_years=10)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)
    
    def test_visualize_feature_analysis(self):
        """Test feature analysis visualization creation."""
        # Create sample correlation dataframe with lag features
        correlation_df = pd.DataFrame({
            'winter_year': range(2015, 2025),
            'winter_severity': [25.0, 22.0, 30.0, 18.0, 28.0, 24.0, 16.0, 26.0, 23.0, 20.0],
            'winter_snowfall': [15.0, 12.5, 20.0, 10.5, 18.0, 14.0, 8.5, 16.0, 13.5, 11.0],
            'winter_temp_avg': [32.0, 33.5, 31.2, 34.0, 32.8, 33.1, 34.5, 32.5, 33.0, 34.2],
            'prev_summer_temp': [70.0, 71.0, 69.0, 72.0, 70.5, 71.5, 73.0, 70.0, 71.0, 72.5],
            'prev_fall_temp': [50.0, 51.0, 49.0, 52.0, 50.5, 51.5, 53.0, 50.0, 51.0, 52.5],
            'prev_winter_severity': [20.0, 25.0, 22.0, 30.0, 18.0, 28.0, 24.0, 16.0, 26.0, 23.0],
            'prev_winter_snowfall': [12.0, 15.0, 12.5, 20.0, 10.5, 18.0, 14.0, 8.5, 16.0, 13.5],
            'prev_winter_temp_avg': [33.0, 32.0, 33.5, 31.2, 34.0, 32.8, 33.1, 34.5, 32.5, 33.0],
            'rolling_2yr_severity': [22.5, 23.5, 26.0, 24.0, 23.0, 22.0, 20.0, 21.0, 24.5, 21.5],
            'rolling_3yr_severity': [22.3, 25.7, 23.3, 25.3, 22.0, 22.7, 22.7, 21.3, 21.7, 23.0],
            'rolling_2yr_snowfall': [13.5, 13.8, 16.3, 15.3, 14.3, 13.0, 11.3, 12.3, 14.8, 12.5],
            'rolling_3yr_snowfall': [13.2, 15.8, 14.3, 16.2, 14.0, 14.2, 13.5, 12.8, 12.7, 13.7],
            'rolling_2yr_temp': [32.8, 32.8, 32.4, 32.6, 33.0, 32.9, 33.8, 33.5, 33.8, 33.6],
            'rolling_3yr_temp': [32.9, 32.2, 32.9, 32.7, 32.6, 33.3, 33.5, 33.4, 33.3, 33.2],
            'enso_oni': [-0.5, 1.0, 0.2, -0.8, 0.1, -0.6, 0.3, 0.9, -0.7, 1.2]
        })
        
        filepath = self.visualizer.visualize_feature_analysis(correlation_df, self.prediction)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 0)


if __name__ == '__main__':
    unittest.main()
