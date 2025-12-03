"""
Weather data fetcher for Boise, Idaho using Open-Meteo API.
Fetches historical daily weather data going back as far as possible.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import json


class WeatherDataFetcher:
    """Fetches historical weather data for Boise, Idaho."""
    
    # Boise, Idaho coordinates
    BOISE_LAT = 43.6150
    BOISE_LON = -116.2023
    
    # Open-Meteo API base URL
    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    # API request timeout in seconds
    REQUEST_TIMEOUT = 60
    
    def __init__(self, data_dir="weather_data"):
        """Initialize the weather data fetcher.
        
        Args:
            data_dir: Directory to store downloaded weather data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def fetch_historical_data(self, start_date="1940-01-01", end_date=None):
        """Fetch historical daily weather data from Open-Meteo API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (default: 1940-01-01)
            end_date: End date in YYYY-MM-DD format (default: yesterday)
            
        Returns:
            pandas.DataFrame: DataFrame with daily weather data
        """
        if end_date is None:
            # Use yesterday as the end date
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        print(f"Fetching weather data from {start_date} to {end_date}...")
        
        # Define weather variables to fetch
        params = {
            "latitude": self.BOISE_LAT,
            "longitude": self.BOISE_LON,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "temperature_2m_max",      # Maximum daily temperature
                "temperature_2m_min",      # Minimum daily temperature
                "temperature_2m_mean",     # Mean daily temperature
                "precipitation_sum",       # Total daily precipitation
                "snowfall_sum",           # Total daily snowfall
                "precipitation_hours",    # Hours of precipitation
                "windspeed_10m_max",      # Maximum wind speed
                "windgusts_10m_max",      # Maximum wind gusts
                "shortwave_radiation_sum", # Solar radiation
                "et0_fao_evapotranspiration" # Evapotranspiration
            ],
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "America/Denver"
        }
        
        # Convert list to comma-separated string
        params["daily"] = ",".join(params["daily"])
        
        try:
            response = requests.get(self.API_URL, params=params, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            daily_data = data["daily"]
            df = pd.DataFrame({
                "date": pd.to_datetime(daily_data["time"]),
                "temp_max": daily_data["temperature_2m_max"],
                "temp_min": daily_data["temperature_2m_min"],
                "temp_mean": daily_data["temperature_2m_mean"],
                "precipitation": daily_data["precipitation_sum"],
                "snowfall": daily_data["snowfall_sum"],
                "precipitation_hours": daily_data["precipitation_hours"],
                "wind_speed_max": daily_data["windspeed_10m_max"],
                "wind_gust_max": daily_data["windgusts_10m_max"],
                "solar_radiation": daily_data["shortwave_radiation_sum"],
                "evapotranspiration": daily_data["et0_fao_evapotranspiration"]
            })
            
            print(f"Successfully fetched {len(df)} days of weather data")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            raise
    
    def save_data(self, df, filename="boise_weather_historical.csv"):
        """Save weather data to CSV file.
        
        Args:
            df: DataFrame with weather data
            filename: Name of the file to save
        """
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
    
    def load_data(self, filename="boise_weather_historical.csv"):
        """Load weather data from CSV file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            pandas.DataFrame: DataFrame with weather data
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        print(f"Loaded {len(df)} days of weather data from {filepath}")
        return df
    
    def fetch_and_save(self, start_date="1940-01-01", end_date=None):
        """Fetch historical data and save to file.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: DataFrame with weather data
        """
        df = self.fetch_historical_data(start_date, end_date)
        self.save_data(df)
        return df


if __name__ == "__main__":
    # Example usage
    fetcher = WeatherDataFetcher()
    df = fetcher.fetch_and_save()
    print("\nData summary:")
    print(df.describe())
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
