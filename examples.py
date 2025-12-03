"""
Example usage of the Boise Weather Analysis modules.
Demonstrates how to use individual components of the application.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from weather_fetcher import WeatherDataFetcher
from weather_analyzer import WeatherAnalyzer
from winter_predictor import WinterPredictor


def example_1_fetch_data():
    """Example 1: Fetch historical weather data."""
    print("=" * 70)
    print("Example 1: Fetching Historical Weather Data")
    print("=" * 70)
    
    fetcher = WeatherDataFetcher()
    
    # Check if data exists
    try:
        df = fetcher.load_data()
        print(f"✓ Loaded existing data: {len(df)} days")
    except FileNotFoundError:
        print("No existing data found. You can fetch it with:")
        print("  df = fetcher.fetch_and_save()")
        return None
    
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


def example_2_analyze_winters(df):
    """Example 2: Analyze historical winters."""
    print("\n" + "=" * 70)
    print("Example 2: Analyzing Historical Winters")
    print("=" * 70)
    
    analyzer = WeatherAnalyzer(df)
    winter_df = analyzer.calculate_winter_features()
    
    print(f"\n✓ Analyzed {len(winter_df)} complete winters")
    
    # Show statistics by severity category
    print("\nWinters by Severity Category:")
    category_counts = winter_df["severity_category"].value_counts()
    for category, count in category_counts.items():
        print(f"  {category:10s}: {count:2d} winters")
    
    # Show coldest and snowiest winters
    print("\n Top 5 Coldest Winters (Avg Temperature):")
    coldest = winter_df.nsmallest(5, "avg_temp")[["winter_label", "avg_temp", "total_snowfall"]]
    for _, row in coldest.iterrows():
        print(f"  {row['winter_label']}: {row['avg_temp']:.1f}°F, {row['total_snowfall']:.2f}\" snow")
    
    print("\nTop 5 Snowiest Winters:")
    snowiest = winter_df.nlargest(5, "total_snowfall")[["winter_label", "total_snowfall", "avg_temp"]]
    for _, row in snowiest.iterrows():
        print(f"  {row['winter_label']}: {row['total_snowfall']:.2f}\" snow, {row['avg_temp']:.1f}°F avg")
    
    return analyzer, winter_df


def example_3_seasonal_correlations(analyzer):
    """Example 3: Analyze seasonal correlations."""
    print("\n" + "=" * 70)
    print("Example 3: Seasonal Correlation Analysis")
    print("=" * 70)
    
    seasonal_df = analyzer.calculate_seasonal_features()
    correlation_results = analyzer.analyze_seasonal_correlations(seasonal_df)
    correlations = correlation_results["correlations"]
    
    print("\nKey Seasonal Correlations:")
    
    for key, value in correlations.items():
        if value is not None:
            # Format the correlation name
            parts = key.split("_to_")
            if len(parts) == 2:
                from_metric = parts[0].replace("_", " ").title()
                to_metric = parts[1].replace("_", " ").title()
                strength = "Strong" if abs(value) > 0.5 else "Moderate" if abs(value) > 0.3 else "Weak"
                direction = "positive" if value > 0 else "negative"
                
                print(f"\n  {from_metric} → {to_metric}:")
                print(f"    Correlation: {value:.3f} ({strength} {direction})")
    
    return correlation_results


def example_4_make_prediction(analyzer, winter_df, correlation_results):
    """Example 4: Make winter predictions."""
    print("\n" + "=" * 70)
    print("Example 4: Predicting Future Winters")
    print("=" * 70)
    
    # Train predictor
    predictor = WinterPredictor()
    predictor.train(correlation_results["correlation_df"])
    
    # Predict Winter 2025-2026
    print("\nPredicting Winter 2025-2026...")
    prediction = predictor.predict_from_current_data(analyzer, target_winter_year=2026)
    
    print(f"\n  Winter: {prediction['winter_label']}")
    print(f"  Predicted Category: {prediction['predicted_category']}")
    print(f"  Confidence: {prediction['confidence'] * 100:.1f}%")
    print(f"  Expected Temperature: {prediction['predicted_avg_temp']:.1f}°F")
    print(f"  Expected Snowfall: {prediction['predicted_snowfall']:.1f}\"")
    
    # Historical comparison
    comparison = predictor.get_historical_comparison(prediction, winter_df)
    
    print(f"\n  Historical Context:")
    print(f"    Severity Percentile: {comparison['percentile_rank']:.0f}th")
    print(f"    Historical Avg Temp: {comparison['historical_avg_temp']:.1f}°F")
    print(f"    Historical Avg Snow: {comparison['historical_avg_snowfall']:.1f}\"")
    
    # Try predicting a different year
    print("\n\nPredicting Winter 2026-2027...")
    prediction_2027 = predictor.predict_from_current_data(analyzer, target_winter_year=2027)
    
    print(f"  Winter: {prediction_2027['winter_label']}")
    print(f"  Predicted Category: {prediction_2027['predicted_category']}")
    print(f"  Expected Temperature: {prediction_2027['predicted_avg_temp']:.1f}°F")


def example_5_recent_trends(analyzer):
    """Example 5: Analyze recent seasonal trends."""
    print("\n" + "=" * 70)
    print("Example 5: Recent Seasonal Trends")
    print("=" * 70)
    
    # Get trends for last 10 years
    recent_summary = analyzer.get_recent_seasons_summary(n_years=10)
    
    print("\nLast 10 Years of Seasonal Patterns:")
    
    for season in ["winter", "spring", "summer", "fall"]:
        if season in recent_summary:
            stats = recent_summary[season]
            print(f"\n  {season.title()}:")
            print(f"    Average Temperature: {stats['avg_temp']:.1f}°F")
            print(f"    Average Precipitation: {stats['total_precip']:.2f}\"")
            print(f"    Temperature Trend: {stats['temp_trend'].title()}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  Boise Weather Analysis - Usage Examples")
    print("=" * 70)
    print("\nThis script demonstrates various ways to use the weather analysis modules.")
    
    # Example 1: Fetch data
    df = example_1_fetch_data()
    
    if df is None:
        print("\n⚠  Please run the main application first to fetch data:")
        print("   python src/boise_weather_app.py")
        print("\nOr generate sample data:")
        print("   python src/generate_sample_data.py")
        return
    
    # Example 2: Analyze winters
    analyzer, winter_df = example_2_analyze_winters(df)
    
    # Example 3: Seasonal correlations
    correlation_results = example_3_seasonal_correlations(analyzer)
    
    # Example 4: Make predictions
    example_4_make_prediction(analyzer, winter_df, correlation_results)
    
    # Example 5: Recent trends
    example_5_recent_trends(analyzer)
    
    print("\n" + "=" * 70)
    print("  Examples Complete")
    print("=" * 70)
    print("\nFor more information, see README.md")
    print("")


if __name__ == "__main__":
    main()
