"""
Tests for the ENSO classifier module.
"""

import unittest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enso_classifier import ENSOClassifier


class TestENSOClassifier(unittest.TestCase):
    """Test cases for ENSOClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = ENSOClassifier()
    
    def test_initialization(self):
        """Test that the classifier initializes correctly."""
        self.assertIsNotNone(self.classifier)
    
    def test_classify_el_nino_year(self):
        """Test classification of a known El Niño year."""
        # 2015 was a strong El Niño year
        result = self.classifier.classify_year(2015)
        self.assertEqual(result["phase"], "El Niño")
        self.assertIsNotNone(result["oni_value"])
        self.assertGreater(result["oni_value"], 0.5)
    
    def test_classify_la_nina_year(self):
        """Test classification of a known La Niña year."""
        # 2011 was a La Niña year
        result = self.classifier.classify_year(2011)
        self.assertEqual(result["phase"], "La Niña")
        self.assertIsNotNone(result["oni_value"])
        self.assertLess(result["oni_value"], -0.5)
    
    def test_classify_neutral_year(self):
        """Test classification of a neutral year."""
        # 2014 was a neutral year
        result = self.classifier.classify_year(2014)
        self.assertEqual(result["phase"], "Neutral")
        self.assertIsNotNone(result["oni_value"])
        self.assertGreaterEqual(result["oni_value"], -0.5)
        self.assertLessEqual(result["oni_value"], 0.5)
    
    def test_classify_unknown_year(self):
        """Test classification of year with no data."""
        result = self.classifier.classify_year(1900)
        self.assertEqual(result["phase"], "Unknown")
        self.assertIsNone(result["oni_value"])
    
    def test_strength_classification(self):
        """Test that strength is properly classified."""
        # 2015 was a very strong El Niño
        result = self.classifier.classify_year(2015)
        self.assertIn(result["strength"], ["Moderate", "Strong"])
        self.assertGreater(result["oni_value"], 1.0)
    
    def test_winter_classification(self):
        """Test winter-specific classification."""
        # Test winter 2015-2016 (strong El Niño)
        result = self.classifier.classify_winter(2016)
        self.assertEqual(result["phase"], "El Niño")
        self.assertIsNotNone(result["oni_value"])
    
    def test_description_format(self):
        """Test that description is properly formatted."""
        result = self.classifier.classify_year(2015)
        self.assertIn("description", result)
        self.assertIsInstance(result["description"], str)
        self.assertGreater(len(result["description"]), 0)
    
    def test_enso_impact_description(self):
        """Test impact descriptions for each phase."""
        el_nino_impact = self.classifier.get_enso_impact_description("El Niño")
        self.assertIsInstance(el_nino_impact, str)
        self.assertGreater(len(el_nino_impact), 0)
        
        la_nina_impact = self.classifier.get_enso_impact_description("La Niña")
        self.assertIsInstance(la_nina_impact, str)
        self.assertGreater(len(la_nina_impact), 0)
        
        neutral_impact = self.classifier.get_enso_impact_description("Neutral")
        self.assertIsInstance(neutral_impact, str)
        self.assertGreater(len(neutral_impact), 0)
    
    def test_get_historical_enso_years(self):
        """Test getting historical ENSO years."""
        # Get all El Niño years
        el_nino_years = self.classifier.get_historical_enso_years(
            phase="El Niño", min_year=2000, max_year=2020
        )
        self.assertIsInstance(el_nino_years, list)
        self.assertGreater(len(el_nino_years), 0)
        
        # Verify each result has required fields
        for year_data in el_nino_years:
            self.assertIn("year", year_data)
            self.assertIn("phase", year_data)
            self.assertEqual(year_data["phase"], "El Niño")
    
    def test_get_all_historical_years(self):
        """Test getting all historical years."""
        all_years = self.classifier.get_historical_enso_years(
            min_year=2010, max_year=2020
        )
        self.assertIsInstance(all_years, list)
        self.assertEqual(len(all_years), 11)  # 2010-2020 inclusive


if __name__ == "__main__":
    unittest.main()
