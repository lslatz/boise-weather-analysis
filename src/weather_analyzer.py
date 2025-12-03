"""
Weather data analyzer for seasonal pattern analysis.
Calculates seasonal features and correlations for winter prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from enso_classifier import ENSOClassifier


class WeatherAnalyzer:
    """Analyzes weather data to identify seasonal patterns and trends."""
    
    # Winter severity calculation weights
    SEVERITY_COLD_TEMP_WEIGHT = 0.5
    SEVERITY_BELOW_20_WEIGHT = 0.3
    SEVERITY_BELOW_32_WEIGHT = 0.1
    SEVERITY_SNOWFALL_WEIGHT = 0.2
    SEVERITY_SNOW_DAYS_WEIGHT = 0.5
    SEVERITY_WIND_WEIGHT = 0.3
    SEVERITY_WIND_THRESHOLD = 15
    SEVERITY_COLD_TEMP_THRESHOLD = 30
    
    # Winter severity category thresholds
    SEVERITY_MILD_THRESHOLD = 15
    SEVERITY_MODERATE_THRESHOLD = 30
    SEVERITY_SEVERE_THRESHOLD = 50
    
    def __init__(self, df):
        """Initialize the analyzer with weather data.
        
        Args:
            df: DataFrame with daily weather data
        """
        self.df = df.copy()
        self.enso_classifier = ENSOClassifier()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data by adding temporal features."""
        self.df["year"] = self.df["date"].dt.year
        self.df["month"] = self.df["date"].dt.month
        self.df["day_of_year"] = self.df["date"].dt.dayofyear
        
        # Define seasons
        # Winter: Dec, Jan, Feb (winter spans two calendar years)
        # Spring: Mar, Apr, May
        # Summer: Jun, Jul, Aug
        # Fall: Sep, Oct, Nov
        self.df["season"] = self.df["month"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall", 10: "Fall", 11: "Fall"
        })
        
        # Winter year: December belongs to the winter of the next year
        # E.g., Dec 2024 is part of Winter 2024-2025
        self.df["winter_year"] = self.df.apply(
            lambda row: row["year"] if row["month"] != 12 else row["year"] + 1,
            axis=1
        )
    
    def calculate_seasonal_features(self):
        """Calculate seasonal aggregate features for each year.
        
        Returns:
            pandas.DataFrame: DataFrame with seasonal features by year
        """
        seasonal_features = []
        
        years = sorted(self.df["year"].unique())
        
        for year in years:
            year_data = self.df[self.df["year"] == year]
            
            if len(year_data) < 300:  # Skip years with insufficient data
                continue
            
            features = {"year": year}
            
            # Calculate features for each season
            for season in ["Winter", "Spring", "Summer", "Fall"]:
                season_data = year_data[year_data["season"] == season]
                
                if len(season_data) > 0:
                    prefix = season.lower()
                    
                    # Temperature features
                    features[f"{prefix}_temp_mean"] = season_data["temp_mean"].mean()
                    features[f"{prefix}_temp_max_avg"] = season_data["temp_max"].mean()
                    features[f"{prefix}_temp_min_avg"] = season_data["temp_min"].mean()
                    features[f"{prefix}_temp_range"] = (
                        season_data["temp_max"].mean() - season_data["temp_min"].mean()
                    )
                    
                    # Precipitation features
                    features[f"{prefix}_precip_total"] = season_data["precipitation"].sum()
                    features[f"{prefix}_precip_days"] = (season_data["precipitation"] > 0.01).sum()
                    features[f"{prefix}_snowfall_total"] = season_data["snowfall"].sum()
                    
                    # Wind features
                    features[f"{prefix}_wind_avg"] = season_data["wind_speed_max"].mean()
                    features[f"{prefix}_wind_max"] = season_data["wind_speed_max"].max()
                    
                    # Solar radiation
                    features[f"{prefix}_solar_avg"] = season_data["solar_radiation"].mean()
                    
                    # Extreme temperature days
                    if season == "Summer":
                        features[f"{prefix}_hot_days"] = (season_data["temp_max"] > 95).sum()
                    elif season == "Winter":
                        features[f"{prefix}_cold_days"] = (season_data["temp_min"] < 20).sum()
                        features[f"{prefix}_snow_days"] = (season_data["snowfall"] > 0.5).sum()
            
            seasonal_features.append(features)
        
        return pd.DataFrame(seasonal_features)
    
    def calculate_winter_features(self):
        """Calculate winter-specific features by winter year.
        
        Returns:
            pandas.DataFrame: DataFrame with winter features
        """
        winter_features = []
        
        # Get winter data (Dec, Jan, Feb)
        winter_data = self.df[self.df["season"] == "Winter"].copy()
        
        winter_years = sorted(winter_data["winter_year"].unique())
        
        for winter_year in winter_years:
            # Get data for this winter (Dec of prev year + Jan/Feb of current year)
            wy_data = winter_data[winter_data["winter_year"] == winter_year]
            
            if len(wy_data) < 75:  # Need at least ~75 days for a complete winter
                continue
            
            features = {
                "winter_year": winter_year,
                "winter_label": f"{winter_year-1}-{winter_year}"
            }
            
            # Temperature features
            features["avg_temp"] = wy_data["temp_mean"].mean()
            features["avg_max_temp"] = wy_data["temp_max"].mean()
            features["avg_min_temp"] = wy_data["temp_min"].mean()
            features["coldest_temp"] = wy_data["temp_min"].min()
            features["warmest_temp"] = wy_data["temp_max"].max()
            features["temp_variability"] = wy_data["temp_mean"].std()
            
            # Cold days
            features["days_below_20"] = (wy_data["temp_min"] < 20).sum()
            features["days_below_32"] = (wy_data["temp_min"] < 32).sum()
            features["days_max_below_32"] = (wy_data["temp_max"] < 32).sum()
            
            # Precipitation and snow
            features["total_precipitation"] = wy_data["precipitation"].sum()
            features["total_snowfall"] = wy_data["snowfall"].sum()
            features["snow_days"] = (wy_data["snowfall"] > 0.5).sum()
            features["heavy_snow_days"] = (wy_data["snowfall"] > 2).sum()
            features["precip_days"] = (wy_data["precipitation"] > 0.1).sum()
            
            # Wind
            features["avg_wind_speed"] = wy_data["wind_speed_max"].mean()
            features["max_wind_gust"] = wy_data["wind_gust_max"].max()
            
            # Classify winter severity
            features["severity_score"] = self._calculate_winter_severity(features)
            features["severity_category"] = self._categorize_winter_severity(
                features["severity_score"]
            )
            
            # Add ENSO classification for this winter
            enso_info = self.enso_classifier.classify_winter(winter_year)
            features["enso_phase"] = enso_info["phase"]
            features["enso_strength"] = enso_info["strength"]
            features["enso_oni"] = enso_info["oni_value"]
            features["enso_description"] = enso_info["description"]
            
            winter_features.append(features)
        
        return pd.DataFrame(winter_features)
    
    def _calculate_winter_severity(self, winter_features):
        """Calculate a winter severity score based on multiple factors.
        
        Args:
            winter_features: Dictionary of winter features
            
        Returns:
            float: Severity score (higher = more severe)
        """
        score = 0.0
        
        # Cold temperature contributes to severity
        if winter_features["avg_temp"] < self.SEVERITY_COLD_TEMP_THRESHOLD:
            score += (self.SEVERITY_COLD_TEMP_THRESHOLD - winter_features["avg_temp"]) * self.SEVERITY_COLD_TEMP_WEIGHT
        
        # More cold days = more severe
        score += winter_features["days_below_20"] * self.SEVERITY_BELOW_20_WEIGHT
        score += winter_features["days_below_32"] * self.SEVERITY_BELOW_32_WEIGHT
        
        # More snow = more severe
        score += winter_features["total_snowfall"] * self.SEVERITY_SNOWFALL_WEIGHT
        score += winter_features["snow_days"] * self.SEVERITY_SNOW_DAYS_WEIGHT
        
        # High wind = more severe
        if winter_features["avg_wind_speed"] > self.SEVERITY_WIND_THRESHOLD:
            score += (winter_features["avg_wind_speed"] - self.SEVERITY_WIND_THRESHOLD) * self.SEVERITY_WIND_WEIGHT
        
        return score
    
    def _categorize_winter_severity(self, score):
        """Categorize winter severity based on score.
        
        Args:
            score: Winter severity score
            
        Returns:
            str: Category (Mild, Moderate, Severe, Extreme)
        """
        if score < self.SEVERITY_MILD_THRESHOLD:
            return "Mild"
        elif score < self.SEVERITY_MODERATE_THRESHOLD:
            return "Moderate"
        elif score < self.SEVERITY_SEVERE_THRESHOLD:
            return "Severe"
        else:
            return "Extreme"
    
    def analyze_seasonal_correlations(self, seasonal_df):
        """Analyze correlations between seasons and following winter.
        
        Args:
            seasonal_df: DataFrame with seasonal features
            
        Returns:
            dict: Dictionary of correlation insights
        """
        # Merge with winter data
        winter_df = self.calculate_winter_features()
        
        # Create lagged features (previous summer/fall predicting next winter)
        merged_data = []
        
        for idx, row in winter_df.iterrows():
            winter_year = row["winter_year"]
            prev_year = winter_year - 1
            
            # Find previous summer and fall data
            prev_seasonal = seasonal_df[seasonal_df["year"] == prev_year]
            
            if len(prev_seasonal) > 0:
                merged_row = {
                    "winter_year": winter_year,
                    "winter_severity": row["severity_score"],
                    "winter_category": row["severity_category"],
                    "winter_temp_avg": row["avg_temp"],
                    "winter_snowfall": row["total_snowfall"],
                }
                
                # Add previous summer features
                if "summer_temp_mean" in prev_seasonal.columns:
                    merged_row["prev_summer_temp"] = prev_seasonal["summer_temp_mean"].values[0]
                    merged_row["prev_summer_precip"] = prev_seasonal["summer_precip_total"].values[0]
                    merged_row["prev_summer_hot_days"] = prev_seasonal["summer_hot_days"].values[0]
                
                # Add previous fall features
                if "fall_temp_mean" in prev_seasonal.columns:
                    merged_row["prev_fall_temp"] = prev_seasonal["fall_temp_mean"].values[0]
                    merged_row["prev_fall_precip"] = prev_seasonal["fall_precip_total"].values[0]
                
                # Add ENSO features for this winter
                enso_info = self.enso_classifier.classify_winter(winter_year)
                merged_row["enso_oni"] = enso_info["oni_value"] if enso_info["oni_value"] is not None else 0.0
                # One-hot encode ENSO phase
                merged_row["enso_el_nino"] = 1.0 if enso_info["phase"] == "El Niño" else 0.0
                merged_row["enso_la_nina"] = 1.0 if enso_info["phase"] == "La Niña" else 0.0
                merged_row["enso_neutral"] = 1.0 if enso_info["phase"] == "Neutral" else 0.0
                
                # Add lag features: previous winter's data
                prev_winter = winter_df[winter_df["winter_year"] == winter_year - 1]
                if len(prev_winter) > 0:
                    merged_row["prev_winter_severity"] = prev_winter["severity_score"].values[0]
                    merged_row["prev_winter_snowfall"] = prev_winter["total_snowfall"].values[0]
                    merged_row["prev_winter_temp_avg"] = prev_winter["avg_temp"].values[0]
                
                merged_data.append(merged_row)
        
        correlation_df = pd.DataFrame(merged_data)
        
        # Add rolling averages (2-year and 3-year)
        if len(correlation_df) > 0:
            # Sort by winter_year to ensure proper rolling calculation
            correlation_df = correlation_df.sort_values("winter_year").reset_index(drop=True)
            
            # 2-year rolling averages
            correlation_df["rolling_2yr_severity"] = correlation_df["winter_severity"].rolling(window=2, min_periods=1).mean()
            correlation_df["rolling_2yr_snowfall"] = correlation_df["winter_snowfall"].rolling(window=2, min_periods=1).mean()
            correlation_df["rolling_2yr_temp"] = correlation_df["winter_temp_avg"].rolling(window=2, min_periods=1).mean()
            
            # 3-year rolling averages
            correlation_df["rolling_3yr_severity"] = correlation_df["winter_severity"].rolling(window=3, min_periods=1).mean()
            correlation_df["rolling_3yr_snowfall"] = correlation_df["winter_snowfall"].rolling(window=3, min_periods=1).mean()
            correlation_df["rolling_3yr_temp"] = correlation_df["winter_temp_avg"].rolling(window=3, min_periods=1).mean()
        
        # Calculate correlations
        correlations = {}
        if len(correlation_df) > 10:
            correlations["summer_temp_to_winter_severity"] = correlation_df[
                ["prev_summer_temp", "winter_severity"]
            ].corr().iloc[0, 1] if "prev_summer_temp" in correlation_df.columns else None
            
            correlations["summer_temp_to_winter_snowfall"] = correlation_df[
                ["prev_summer_temp", "winter_snowfall"]
            ].corr().iloc[0, 1] if "prev_summer_temp" in correlation_df.columns else None
            
            correlations["fall_temp_to_winter_temp"] = correlation_df[
                ["prev_fall_temp", "winter_temp_avg"]
            ].corr().iloc[0, 1] if "prev_fall_temp" in correlation_df.columns else None
            
            correlations["enso_oni_to_winter_severity"] = correlation_df[
                ["enso_oni", "winter_severity"]
            ].corr().iloc[0, 1] if "enso_oni" in correlation_df.columns else None
            
            correlations["enso_oni_to_winter_temp"] = correlation_df[
                ["enso_oni", "winter_temp_avg"]
            ].corr().iloc[0, 1] if "enso_oni" in correlation_df.columns else None
            
            # Lag feature correlations
            correlations["prev_winter_severity_to_winter_severity"] = correlation_df[
                ["prev_winter_severity", "winter_severity"]
            ].corr().iloc[0, 1] if "prev_winter_severity" in correlation_df.columns else None
            
            correlations["prev_winter_snowfall_to_winter_snowfall"] = correlation_df[
                ["prev_winter_snowfall", "winter_snowfall"]
            ].corr().iloc[0, 1] if "prev_winter_snowfall" in correlation_df.columns else None
            
            correlations["rolling_2yr_severity_to_winter_severity"] = correlation_df[
                ["rolling_2yr_severity", "winter_severity"]
            ].corr().iloc[0, 1] if "rolling_2yr_severity" in correlation_df.columns else None
            
            correlations["rolling_3yr_severity_to_winter_severity"] = correlation_df[
                ["rolling_3yr_severity", "winter_severity"]
            ].corr().iloc[0, 1] if "rolling_3yr_severity" in correlation_df.columns else None
        
        return {
            "correlations": correlations,
            "correlation_df": correlation_df,
            "winter_df": winter_df  # Add winter_df for visualization purposes
        }
    
    def get_recent_seasons_summary(self, n_years=5):
        """Get summary of recent seasons for context.
        
        Args:
            n_years: Number of recent years to summarize
            
        Returns:
            dict: Summary of recent seasonal patterns
        """
        current_year = datetime.now().year
        recent_years = range(current_year - n_years, current_year)
        
        recent_data = self.df[self.df["year"].isin(recent_years)]
        
        summary = {}
        for season in ["Winter", "Spring", "Summer", "Fall"]:
            season_data = recent_data[recent_data["season"] == season]
            if len(season_data) > 0:
                summary[season.lower()] = {
                    "avg_temp": season_data["temp_mean"].mean(),
                    "total_precip": season_data["precipitation"].sum() / n_years,
                    "temp_trend": "warming" if season_data.groupby("year")["temp_mean"].mean().diff().mean() > 0 else "cooling"
                }
        
        return summary
