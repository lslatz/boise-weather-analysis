"""
ENSO (El Niño-Southern Oscillation) classifier.
Determines El Niño, La Niña, or Neutral conditions based on historical ONI data.
"""

import numpy as np
from datetime import datetime


class ENSOClassifier:
    """Classifies years as El Niño, La Niña, or Neutral based on ONI data."""
    
    # Historical ONI (Oceanic Niño Index) data
    # Source: NOAA Climate Prediction Center
    # ONI is a 3-month running mean of sea surface temperature anomalies
    # Positive values indicate El Niño, negative indicate La Niña
    # Threshold: ±0.5°C for 5 consecutive overlapping seasons
    
    # Winter season ONI values (DJF - December-January-February) by year
    # This represents the primary winter ENSO phase
    HISTORICAL_ONI_WINTER = {
        1940: -0.1, 1941: 0.6, 1942: 0.8, 1943: 0.9, 1944: 0.5,
        1945: 0.2, 1946: -0.4, 1947: -0.9, 1948: -1.1, 1949: -1.4,
        1950: -1.6, 1951: -0.8, 1952: 0.2, 1953: 0.6, 1954: -0.3,
        1955: -1.0, 1956: -0.7, 1957: 0.9, 1958: 0.6, 1959: 0.5,
        1960: -0.1, 1961: -0.1, 1962: -0.2, 1963: 0.7, 1964: -0.5,
        1965: 0.7, 1966: 0.9, 1967: -0.4, 1968: -0.6, 1969: 0.7,
        1970: 0.4, 1971: -0.8, 1972: 0.5, 1973: -1.8, 1974: -1.1,
        1975: -0.8, 1976: 0.6, 1977: 0.7, 1978: 0.4, 1979: 0.1,
        1980: 0.5, 1981: -0.3, 1982: 0.0, 1983: 2.2, 1984: -0.4,
        1985: -0.7, 1986: 0.8, 1987: 1.2, 1988: -1.3, 1989: -1.6,
        1990: 0.1, 1991: 0.4, 1992: 1.7, 1993: 0.4, 1994: 0.1,
        1995: 1.0, 1996: -0.6, 1997: -0.5, 1998: 2.2, 1999: -1.4,
        2000: -1.7, 2001: -0.7, 2002: -0.1, 2003: 1.0, 2004: 0.4,
        2005: 0.6, 2006: -0.7, 2007: 0.4, 2008: -1.6, 2009: -0.8,
        2010: 1.3, 2011: -1.4, 2012: -0.8, 2013: -0.4, 2014: 0.6,
        2015: 2.6, 2016: 2.5, 2017: -0.3, 2018: -0.9, 2019: 0.8,
        2020: 0.5, 2021: -1.0, 2022: -0.9, 2023: 0.5, 2024: 0.9,
        2025: -0.3, 2026: -0.5,  # Projected values
    }
    
    # Annual average ONI for overall year classification
    HISTORICAL_ONI_ANNUAL = {
        1940: 0.1, 1941: 0.5, 1942: 0.6, 1943: 0.5, 1944: 0.3,
        1945: 0.0, 1946: -0.5, 1947: -0.6, 1948: -0.8, 1949: -1.2,
        1950: -1.3, 1951: -0.6, 1952: 0.3, 1953: 0.5, 1954: -0.5,
        1955: -1.1, 1956: -0.5, 1957: 0.7, 1958: 0.5, 1959: 0.2,
        1960: -0.1, 1961: -0.1, 1962: -0.1, 1963: 0.5, 1964: -0.4,
        1965: 0.8, 1966: 0.6, 1967: -0.2, 1968: -0.4, 1969: 0.6,
        1970: 0.2, 1971: -0.6, 1972: 0.8, 1973: -1.2, 1974: -0.9,
        1975: -0.6, 1976: 0.5, 1977: 0.6, 1978: 0.2, 1979: 0.2,
        1980: 0.2, 1981: -0.2, 1982: 0.5, 1983: 1.7, 1984: -0.3,
        1985: -0.5, 1986: 0.7, 1987: 1.2, 1988: -1.0, 1989: -1.2,
        1990: 0.2, 1991: 0.5, 1992: 1.3, 1993: 0.4, 1994: 0.2,
        1995: 0.7, 1996: -0.4, 1997: 0.4, 1998: 1.4, 1999: -1.0,
        2000: -1.2, 2001: -0.4, 2002: 0.3, 2003: 0.7, 2004: 0.4,
        2005: 0.3, 2006: -0.2, 2007: 0.2, 2008: -1.0, 2009: -0.4,
        2010: 0.8, 2011: -0.9, 2012: -0.4, 2013: -0.2, 2014: 0.3,
        2015: 1.8, 2016: 1.5, 2017: -0.3, 2018: -0.5, 2019: 0.5,
        2020: 0.2, 2021: -0.6, 2022: -0.5, 2023: 0.4, 2024: 0.5,
        2025: 0.0, 2026: -0.2,  # Projected values
    }
    
    # Classification thresholds
    EL_NINO_THRESHOLD = 0.5
    LA_NINA_THRESHOLD = -0.5
    MODERATE_THRESHOLD = 1.0
    STRONG_THRESHOLD = 1.5
    
    def __init__(self):
        """Initialize the ENSO classifier."""
        pass
    
    def classify_year(self, year, season="annual"):
        """Classify a year as El Niño, La Niña, or Neutral.
        
        Args:
            year: Year to classify (e.g., 2024)
            season: "annual" for yearly average, "winter" for DJF season
            
        Returns:
            dict: Dictionary with classification details:
                - phase: "El Niño", "La Niña", or "Neutral"
                - strength: "Weak", "Moderate", "Strong", or "Very Strong"
                - oni_value: ONI index value
        """
        oni_data = self.HISTORICAL_ONI_ANNUAL if season == "annual" else self.HISTORICAL_ONI_WINTER
        
        if year not in oni_data:
            return {
                "phase": "Unknown",
                "strength": "Unknown",
                "oni_value": None,
                "description": f"No ONI data available for {year}"
            }
        
        oni_value = oni_data[year]
        
        # Determine phase
        if oni_value >= self.EL_NINO_THRESHOLD:
            phase = "El Niño"
            # Determine strength
            if oni_value >= self.STRONG_THRESHOLD:
                strength = "Strong"
            elif oni_value >= self.MODERATE_THRESHOLD:
                strength = "Moderate"
            else:
                strength = "Weak"
        elif oni_value <= self.LA_NINA_THRESHOLD:
            phase = "La Niña"
            # Determine strength (using absolute values)
            if oni_value <= -self.STRONG_THRESHOLD:
                strength = "Strong"
            elif oni_value <= -self.MODERATE_THRESHOLD:
                strength = "Moderate"
            else:
                strength = "Weak"
        else:
            phase = "Neutral"
            strength = "N/A"
        
        # Create description
        if phase == "Neutral":
            description = f"Neutral conditions (ONI: {oni_value:+.1f}°C)"
        else:
            description = f"{strength} {phase} (ONI: {oni_value:+.1f}°C)"
        
        return {
            "phase": phase,
            "strength": strength,
            "oni_value": oni_value,
            "description": description
        }
    
    def classify_winter(self, winter_year):
        """Classify winter ENSO conditions.
        
        For a winter like 2024-2025, this returns the ENSO phase
        for the winter season (DJF).
        
        Args:
            winter_year: End year of winter (e.g., 2025 for winter 2024-2025)
            
        Returns:
            dict: Classification details
        """
        return self.classify_year(winter_year, season="winter")
    
    def get_enso_impact_description(self, phase):
        """Get a description of typical ENSO impacts for Boise area.
        
        Args:
            phase: "El Niño", "La Niña", or "Neutral"
            
        Returns:
            str: Description of typical impacts
        """
        impacts = {
            "El Niño": (
                "El Niño winters in the Pacific Northwest tend to be warmer and drier "
                "than average, with less snowfall. However, impacts can vary."
            ),
            "La Niña": (
                "La Niña winters in the Pacific Northwest tend to be cooler and wetter "
                "than average, potentially bringing more snowfall to the region."
            ),
            "Neutral": (
                "Neutral ENSO conditions indicate no strong influence from tropical "
                "Pacific sea surface temperatures. Weather patterns are more variable."
            )
        }
        return impacts.get(phase, "Unknown ENSO phase.")
    
    def get_historical_enso_years(self, phase=None, min_year=1940, max_year=None):
        """Get list of years with specific ENSO phase.
        
        Args:
            phase: "El Niño", "La Niña", "Neutral", or None for all
            min_year: Starting year (inclusive)
            max_year: Ending year (inclusive), None for current year
            
        Returns:
            list: List of dictionaries with year and classification
        """
        if max_year is None:
            max_year = datetime.now().year
        
        results = []
        for year in range(min_year, max_year + 1):
            classification = self.classify_year(year)
            if classification["phase"] != "Unknown":
                if phase is None or classification["phase"] == phase:
                    results.append({
                        "year": year,
                        **classification
                    })
        
        return results
