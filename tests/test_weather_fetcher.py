"""
Tests for the weather fetcher module.
"""

import unittest
import pandas as pd
import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from weather_fetcher import WeatherDataFetcher


class TestWeatherDataFetcher(unittest.TestCase):
    """Test cases for WeatherDataFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.fetcher = WeatherDataFetcher(data_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test that the fetcher initializes correctly."""
        self.assertEqual(self.fetcher.data_dir, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))
    
    def test_save_and_load_data(self):
        """Test saving and loading weather data."""
        # Create sample data
        dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq='D')
        sample_df = pd.DataFrame({
            "date": dates,
            "temp_max": [50.0] * len(dates),
            "temp_min": [30.0] * len(dates),
            "temp_mean": [40.0] * len(dates),
            "precipitation": [0.1] * len(dates),
            "snowfall": [0.0] * len(dates),
            "precipitation_hours": [2] * len(dates),
            "wind_speed_max": [10.0] * len(dates),
            "wind_gust_max": [15.0] * len(dates),
            "solar_radiation": [300.0] * len(dates),
            "evapotranspiration": [0.1] * len(dates)
        })
        
        # Save data
        filename = "test_weather.csv"
        self.fetcher.save_data(sample_df, filename)
        
        # Load data
        loaded_df = self.fetcher.load_data(filename)
        
        # Verify
        self.assertEqual(len(loaded_df), len(sample_df))
        self.assertTrue("date" in loaded_df.columns)
        pd.testing.assert_frame_equal(
            loaded_df.reset_index(drop=True), 
            sample_df.reset_index(drop=True)
        )
    
    def test_coordinates(self):
        """Test that Boise coordinates are correct."""
        # Boise, Idaho approximate coordinates
        self.assertAlmostEqual(self.fetcher.BOISE_LAT, 43.6150, places=2)
        self.assertAlmostEqual(self.fetcher.BOISE_LON, -116.2023, places=2)


if __name__ == '__main__':
    unittest.main()
