"""
Generate sample weather data for testing when API is unavailable.
Creates realistic synthetic weather data for Boise, Idaho.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sample_data(start_date="1940-01-01", end_date="2024-12-02", output_dir="weather_data"):
    """Generate sample weather data for testing.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_dir: Directory to save the sample data
        
    Returns:
        pandas.DataFrame: Sample weather data
    """
    print("Generating sample weather data...")
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Initialize lists for data
    temp_max = []
    temp_min = []
    temp_mean = []
    precipitation = []
    snowfall = []
    precip_hours = []
    wind_speed = []
    wind_gust = []
    solar_rad = []
    evap = []
    
    # Generate data with seasonal patterns
    for i, date in enumerate(dates):
        month = date.month
        day_of_year = date.dayofyear
        year = date.year
        
        # Base temperature varies by season (sinusoidal pattern)
        base_temp = 55 + 30 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Add long-term warming trend (about 0.5°F per decade since 1940)
        years_since_1940 = year - 1940
        warming_trend = years_since_1940 * 0.05
        
        # Daily temperature range
        daily_range = 25 + 10 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Add random variation
        random_variation = np.random.normal(0, 5)
        
        # Calculate temperatures
        t_mean = base_temp + warming_trend + random_variation
        t_max = t_mean + daily_range / 2 + np.random.normal(0, 3)
        t_min = t_mean - daily_range / 2 + np.random.normal(0, 3)
        
        # Precipitation (higher in spring and winter)
        precip_likelihood = 0.15 + 0.1 * np.sin(2 * np.pi * (day_of_year - 120) / 365)
        has_precip = np.random.random() < precip_likelihood
        precip = np.random.exponential(0.2) if has_precip else 0.0
        
        # Snowfall (only when cold and precipitating)
        snow = 0.0
        if has_precip and t_mean < 35:
            snow_fraction = (35 - t_mean) / 15  # More snow when colder
            snow_fraction = min(1.0, max(0.0, snow_fraction))
            snow = precip * snow_fraction * 10  # Snow:rain ratio roughly 10:1
        
        # Precipitation hours
        p_hours = 0
        if has_precip:
            p_hours = np.random.randint(1, 12)
        
        # Wind (higher in spring and fall)
        wind_base = 8 + 5 * np.sin(2 * np.pi * (day_of_year - 150) / 365)
        wind = max(0, wind_base + np.random.normal(0, 4))
        gust = wind * (1.3 + np.random.random() * 0.4)
        
        # Solar radiation (higher in summer)
        solar = max(0, 300 + 200 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 50))
        
        # Evapotranspiration (higher in summer)
        evapotrans = max(0, 0.15 + 0.1 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 0.03))
        
        # Append to lists
        temp_max.append(round(t_max, 1))
        temp_min.append(round(t_min, 1))
        temp_mean.append(round(t_mean, 1))
        precipitation.append(round(precip, 2))
        snowfall.append(round(snow, 2))
        precip_hours.append(p_hours)
        wind_speed.append(round(wind, 1))
        wind_gust.append(round(gust, 1))
        solar_rad.append(round(solar, 1))
        evap.append(round(evapotrans, 3))
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "temp_mean": temp_mean,
        "precipitation": precipitation,
        "snowfall": snowfall,
        "precipitation_hours": precip_hours,
        "wind_speed_max": wind_speed,
        "wind_gust_max": wind_gust,
        "solar_radiation": solar_rad,
        "evapotranspiration": evap
    })
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "boise_weather_historical.csv")
    df.to_csv(filepath, index=False)
    
    print(f"Sample data generated: {len(df)} days")
    print(f"Saved to: {filepath}")
    
    return df


if __name__ == "__main__":
    df = generate_sample_data()
    print("\nSample statistics:")
    print(df.describe())
