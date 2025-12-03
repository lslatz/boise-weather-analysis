# Boise Weather Analysis & Winter Prediction

A Python application that analyzes historical weather patterns in Boise, Idaho to predict future winter conditions based on seasonal correlations.

## Overview

This application:
- Fetches comprehensive historical weather data from the Open-Meteo API (going back to 1940)
- Analyzes daily weather data including temperature, precipitation, snowfall, wind, and solar radiation
- Calculates seasonal features and patterns
- Identifies correlations between seasons (e.g., how summer conditions influence the following winter)
- Uses machine learning to predict winter severity, snowfall, and temperature
- Provides predictions for Winter 2025-2026 (and can be adapted for future seasons)

## Features

### Data Collection
- **Historical Data**: Fetches daily weather data from 1940 to present
- **Comprehensive Metrics**: Temperature (max/min/mean), precipitation, snowfall, wind speed, solar radiation, and evapotranspiration
- **Location**: Boise, Idaho (43.6150°N, -116.2023°W)

### Analysis Capabilities
- **Seasonal Aggregation**: Calculates features for Winter, Spring, Summer, and Fall
- **Winter Classification**: Categorizes winters as Mild, Moderate, Severe, or Extreme
- **ENSO Detection**: Identifies El Niño, La Niña, or Neutral conditions for each winter year
- **Correlation Analysis**: Identifies relationships between seasonal patterns
- **Trend Detection**: Tracks warming/cooling trends over recent years

### Prediction Model
- **Multi-factor Prediction**: Uses Random Forest models to predict winter conditions
- **Key Predictions**:
  - Winter severity category (Mild/Moderate/Severe/Extreme)
  - Average winter temperature
  - Total snowfall
  - Confidence scores and probability distributions
- **Historical Comparison**: Compares predictions to similar past winters

### Visualizations
- **Prediction Charts**: Comprehensive visualization of winter predictions including:
  - Category probabilities with confidence levels
  - Predicted vs historical temperature and snowfall comparison
  - Severity score gauge showing prediction on severity scale
  - Input features used for prediction
- **Historical Trends**: Last 10 years of winter data visualized with:
  - Temperature trends (average, max, min)
  - Snowfall patterns by year
  - Severity scores over time
  - Distribution of winter severity categories
- **Automatic Generation**: Visualizations are automatically saved as PNG files in the `visualizations/` directory

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for fetching weather data)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lslatz/boise-weather-analysis.git
cd boise-weather-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Simply run the main application:

```bash
python src/boise_weather_app.py
```

The application will:
1. Fetch historical weather data (or load existing data if available)
2. Analyze seasonal patterns and correlations
3. Train prediction models on historical data
4. Generate a prediction for Winter 2025-2026
5. Display insights about seasonal correlations and recent trends
6. **Generate and save visualizations** showing prediction details and historical trends

### Example Output

```
======================================================================
  Boise Weather Analysis & Winter Prediction
======================================================================

This application analyzes historical weather patterns in Boise, Idaho
to predict winter conditions based on preceding seasonal data.

✓ Loaded 30,500 days of weather data
  Date range: January 01, 1940 to December 02, 2024

======================================================================
  Analyzing Weather Patterns
======================================================================

✓ Analyzed 84 years of seasonal data
✓ Analyzed 83 complete winters

  Recent Winters:
    2020-2021: Moderate    (Avg: 32.5°F, Snow: 18.35") [La Niña]
    2021-2022: Mild        (Avg: 35.2°F, Snow: 12.10") [La Niña]
    2022-2023: Severe      (Avg: 28.8°F, Snow: 25.60") [El Niño]
    2023-2024: Moderate    (Avg: 31.9°F, Snow: 19.45") [El Niño]
    2024-2025: Mild        (Avg: 34.1°F, Snow: 14.20") [Neutral]

======================================================================
  Winter 2025-2026 Prediction
======================================================================

Based on weather patterns from 2025:

  ENSO Conditions: Weak La Niña (ONI: -0.5°C)
  La Niña winters in the Pacific Northwest tend to be cooler and wetter
  than average, potentially bringing more snowfall to the region.

  Predicted Category: Moderate
  Confidence: 68.5%

  Predicted Average Temperature: 32.1°F
  Predicted Total Snowfall: 19.25"
  Severity Score: 24.3

  Category Probabilities:
    Moderate   [████████████████████        ] 68.5%
    Severe     [████████                    ] 22.1%
    Mild       [██                          ]  9.4%

======================================================================
  Generating Visualizations
======================================================================

✓ Prediction visualization saved: visualizations/winter_2025-2026_prediction.png
✓ Historical winters visualization saved: visualizations/historical_winters_last_10_years.png

  Visualizations include:
    • Winter 2025-2026 prediction details
    • Category probabilities and severity score
    • Temperature and snowfall predictions
    • Last 10 years of historical winter data
    • Temperature trends, snowfall patterns, and severity distribution
```

### Using Individual Modules

You can also use the individual modules in your own scripts:

#### Fetch Weather Data
```python
from src.weather_fetcher import WeatherDataFetcher

fetcher = WeatherDataFetcher()
df = fetcher.fetch_and_save(start_date="1940-01-01")
print(df.head())
```

#### Analyze Patterns
```python
from src.weather_analyzer import WeatherAnalyzer

analyzer = WeatherAnalyzer(df)
seasonal_df = analyzer.calculate_seasonal_features()
winter_df = analyzer.calculate_winter_features()

# Winter features now include ENSO classification
print(winter_df[['winter_label', 'severity_category', 'enso_phase', 'enso_description']].tail())
```

#### Check ENSO Conditions
```python
from src.enso_classifier import ENSOClassifier

enso = ENSOClassifier()

# Classify a specific year
enso_2024 = enso.classify_year(2024)
print(f"2024: {enso_2024['description']}")

# Get winter-specific classification
winter_2026 = enso.classify_winter(2026)
print(f"Winter 2025-2026: {winter_2026['phase']}")

# Get impact description
impact = enso.get_enso_impact_description(winter_2026['phase'])
print(impact)

# Get all El Niño years since 2000
el_nino_years = enso.get_historical_enso_years(phase="El Niño", min_year=2000, max_year=2024)
for year_data in el_nino_years:
    print(f"{year_data['year']}: {year_data['description']}")
```

#### Make Predictions
```python
from src.winter_predictor import WinterPredictor

predictor = WinterPredictor()
correlation_results = analyzer.analyze_seasonal_correlations(seasonal_df)
predictor.train(correlation_results['correlation_df'])

prediction = predictor.predict_from_current_data(analyzer, target_winter_year=2026)
print(f"Predicted winter category: {prediction['predicted_category']}")
```

#### Generate Visualizations
```python
from src.weather_visualizer import WeatherVisualizer

visualizer = WeatherVisualizer()

# Create prediction visualization
prediction_path = visualizer.visualize_prediction(prediction, winter_df)
print(f"Prediction chart saved to: {prediction_path}")

# Create historical winters visualization
historical_path = visualizer.visualize_historical_winters(winter_df, n_years=10)
print(f"Historical chart saved to: {historical_path}")

# Or create both at once
pred_path, hist_path = visualizer.visualize_both(prediction, winter_df, n_years=10)
```

## How It Works

### 1. Data Collection
The application uses the Open-Meteo Historical Weather API to fetch daily weather observations for Boise. This free API provides:
- Temperature data (max, min, mean)
- Precipitation and snowfall measurements
- Wind speed and gusts
- Solar radiation
- Evapotranspiration

### 2. Feature Engineering
The analyzer calculates aggregate features for each season:
- Average, maximum, and minimum temperatures
- Total precipitation and snowfall
- Number of extreme weather days
- Wind patterns
- Solar radiation averages

### 3. Winter Severity Classification
Each winter is assigned a severity score based on:
- Average and extreme temperatures
- Number of cold days (below 20°F and 32°F)
- Total snowfall and snow days
- Wind patterns

Winters are then categorized as:
- **Mild**: Warm temperatures, minimal snow
- **Moderate**: Average conditions
- **Severe**: Cold temperatures, significant snowfall
- **Extreme**: Exceptionally cold and snowy

### 4. ENSO (El Niño-Southern Oscillation) Classification
Each winter is classified based on the Oceanic Niño Index (ONI):
- **El Niño**: Warmer Pacific sea surface temperatures (ONI ≥ +0.5°C)
  - Tends to bring warmer, drier winters to the Pacific Northwest
- **La Niña**: Cooler Pacific sea surface temperatures (ONI ≤ -0.5°C)
  - Tends to bring cooler, wetter winters with more snowfall
- **Neutral**: Normal conditions (-0.5°C < ONI < +0.5°C)
  - No strong ENSO influence on winter weather patterns

The application uses historical ONI data from NOAA to identify ENSO conditions for each winter season.

### 5. Correlation Analysis
The system analyzes relationships between:
- Summer temperature → Winter severity
- Summer heat waves → Winter snowfall
- Fall precipitation → Winter conditions
- Long-term seasonal trends

### 6. Prediction
Random Forest machine learning models predict:
- Winter severity score
- Expected snowfall
- Average temperature
- Category classification with confidence scores

The models learn from 80+ years of historical data to identify patterns that persist across decades.

## Customization

### Predicting Different Years
To predict a different winter, modify the target year in `boise_weather_app.py`:

```python
prediction = predictor.predict_from_current_data(analyzer, target_winter_year=2027)
```

### Analyzing Different Locations
To analyze a different location, update the coordinates in `src/weather_fetcher.py`:

```python
LAT = 40.7608  # Example: Salt Lake City
LON = -111.8910
```

### Adjusting Date Ranges
To fetch data for a different time period:

```python
df = fetcher.fetch_and_save(start_date="1950-01-01", end_date="2020-12-31")
```

## Data Sources

This application uses:
- **[Open-Meteo Historical Weather API](https://open-meteo.com/)**: 
  - Free and open-source weather data
  - No API key required
  - Historical data from 1940 onwards
  - High-quality reanalysis data
- **NOAA Climate Prediction Center**:
  - Oceanic Niño Index (ONI) data for ENSO classification
  - Historical ENSO phase information from 1940 to present

## Limitations

- Predictions are based on statistical patterns and correlations, not physical weather models
- Long-term accuracy depends on climate stability (climate change may affect pattern reliability)
- Requires complete seasonal data from the preceding year for best predictions
- Local microclimates and extreme weather events may not be fully captured
- ENSO conditions provide general trends but don't guarantee specific outcomes for any location

## Future Enhancements

Potential improvements:
- Add support for other cities and regions
- Incorporate additional data sources (e.g., NOAA, local weather stations)
- Implement ensemble models combining multiple prediction approaches
- ~~Add visualization of trends and predictions~~ ✓ **Completed**
- ~~Include more sophisticated climate indicators (e.g., El Niño/La Niña)~~ ✓ **Completed**
- Create a web interface for easier interaction
- Add interactive visualizations with filtering and zooming capabilities

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available for use and modification.

## Acknowledgments

- Weather data provided by [Open-Meteo](https://open-meteo.com/)
- Machine learning models built with [scikit-learn](https://scikit-learn.org/)
- Data analysis powered by [pandas](https://pandas.pydata.org/)