"""
Tests for the weather analyzer module.
"""

import unittest
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from weather_analyzer import WeatherAnalyzer


class TestWeatherAnalyzer(unittest.TestCase):
    """Test cases for WeatherAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample weather data spanning multiple years
        dates = []
        start_date = datetime(2020, 1, 1)
        for i in range(365 * 3):  # 3 years of data
            dates.append(start_date + timedelta(days=i))
        
        self.sample_df = pd.DataFrame({
            "date": dates,
            "temp_max": [50.0 + 20 * (i % 365) / 365 for i in range(len(dates))],
            "temp_min": [30.0 + 20 * (i % 365) / 365 for i in range(len(dates))],
            "temp_mean": [40.0 + 20 * (i % 365) / 365 for i in range(len(dates))],
            "precipitation": [0.1] * len(dates),
            "snowfall": [0.5 if i % 365 < 60 or i % 365 > 330 else 0.0 for i in range(len(dates))],
            "precipitation_hours": [2] * len(dates),
            "wind_speed_max": [10.0] * len(dates),
            "wind_gust_max": [15.0] * len(dates),
            "solar_radiation": [300.0] * len(dates),
            "evapotranspiration": [0.1] * len(dates)
        })
        
        self.analyzer = WeatherAnalyzer(self.sample_df)
    
    def test_initialization(self):
        """Test that the analyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer.df)
        self.assertTrue("year" in self.analyzer.df.columns)
        self.assertTrue("month" in self.analyzer.df.columns)
        self.assertTrue("season" in self.analyzer.df.columns)
    
    def test_season_assignment(self):
        """Test that seasons are assigned correctly."""
        # Check winter months
        winter_df = self.analyzer.df[self.analyzer.df["season"] == "Winter"]
        winter_months = winter_df["month"].unique()
        self.assertTrue(all(m in [12, 1, 2] for m in winter_months))
        
        # Check summer months
        summer_df = self.analyzer.df[self.analyzer.df["season"] == "Summer"]
        summer_months = summer_df["month"].unique()
        self.assertTrue(all(m in [6, 7, 8] for m in summer_months))
    
    def test_calculate_seasonal_features(self):
        """Test seasonal feature calculation."""
        seasonal_df = self.analyzer.calculate_seasonal_features()
        
        # Should have data for 3 years
        self.assertGreaterEqual(len(seasonal_df), 2)
        
        # Check for expected columns
        expected_cols = ["year", "summer_temp_mean", "winter_temp_mean", 
                        "spring_temp_mean", "fall_temp_mean"]
        for col in expected_cols:
            self.assertTrue(col in seasonal_df.columns, f"Missing column: {col}")
    
    def test_calculate_winter_features(self):
        """Test winter-specific feature calculation."""
        winter_df = self.analyzer.calculate_winter_features()
        
        # Should have data for multiple winters
        self.assertGreater(len(winter_df), 0)
        
        # Check for expected columns
        expected_cols = ["winter_year", "avg_temp", "total_snowfall", 
                        "severity_score", "severity_category"]
        for col in expected_cols:
            self.assertTrue(col in winter_df.columns, f"Missing column: {col}")
        
        # Check that severity categories are valid
        valid_categories = ["Mild", "Moderate", "Severe", "Extreme"]
        for category in winter_df["severity_category"].unique():
            self.assertTrue(category in valid_categories)
    
    def test_winter_severity_calculation(self):
        """Test winter severity score calculation."""
        winter_features = {
            "avg_temp": 25.0,  # Cold
            "days_below_20": 20,  # Many cold days
            "days_below_32": 50,
            "total_snowfall": 30.0,  # Significant snow
            "snow_days": 25,
            "avg_wind_speed": 12.0
        }
        
        score = self.analyzer._calculate_winter_severity(winter_features)
        
        # Should have a positive severity score for cold, snowy conditions
        self.assertGreater(score, 0)
    
    def test_winter_severity_categorization(self):
        """Test winter severity categorization."""
        # Test mild winter
        self.assertEqual(self.analyzer._categorize_winter_severity(10), "Mild")
        
        # Test moderate winter
        self.assertEqual(self.analyzer._categorize_winter_severity(20), "Moderate")
        
        # Test severe winter
        self.assertEqual(self.analyzer._categorize_winter_severity(40), "Severe")
        
        # Test extreme winter
        self.assertEqual(self.analyzer._categorize_winter_severity(60), "Extreme")
    
    def test_lag_features_in_correlation_analysis(self):
        """Test that lag features are included in correlation analysis."""
        seasonal_df = self.analyzer.calculate_seasonal_features()
        correlation_results = self.analyzer.analyze_seasonal_correlations(seasonal_df)
        correlation_df = correlation_results["correlation_df"]
        
        # Check for lag features
        if len(correlation_df) > 1:  # Need at least 2 winters for lag features
            self.assertTrue("prev_winter_severity" in correlation_df.columns or 
                          len(correlation_df) == 1,
                          "Previous winter severity should be in correlation data")
            self.assertTrue("prev_winter_snowfall" in correlation_df.columns or 
                          len(correlation_df) == 1,
                          "Previous winter snowfall should be in correlation data")
    
    def test_rolling_averages_in_correlation_analysis(self):
        """Test that rolling averages are calculated."""
        seasonal_df = self.analyzer.calculate_seasonal_features()
        correlation_results = self.analyzer.analyze_seasonal_correlations(seasonal_df)
        correlation_df = correlation_results["correlation_df"]
        
        # Check for rolling average features
        if len(correlation_df) > 0:
            self.assertTrue("rolling_2yr_severity" in correlation_df.columns,
                          "2-year rolling average severity should be present")
            self.assertTrue("rolling_3yr_severity" in correlation_df.columns,
                          "3-year rolling average severity should be present")
            self.assertTrue("rolling_2yr_snowfall" in correlation_df.columns,
                          "2-year rolling average snowfall should be present")


if __name__ == '__main__':
    unittest.main()
