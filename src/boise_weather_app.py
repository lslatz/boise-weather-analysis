"""
Main application for Boise Winter Weather Prediction.
Fetches data, analyzes patterns, and predicts future winter conditions.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from weather_fetcher import WeatherDataFetcher
from weather_analyzer import WeatherAnalyzer
from winter_predictor import WinterPredictor


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def format_temp(temp):
    """Format temperature with degree symbol."""
    return f"{temp:.1f}°F"


def format_precip(precip):
    """Format precipitation in inches."""
    return f"{precip:.2f}\""


def main():
    """Main application entry point."""
    print_section("Boise Weather Analysis & Winter Prediction")
    print("\nThis application analyzes historical weather patterns in Boise, Idaho")
    print("to predict winter conditions based on preceding seasonal data.")
    
    # Initialize components
    fetcher = WeatherDataFetcher()
    
    # Check if data already exists
    data_file = os.path.join(fetcher.data_dir, "boise_weather_historical.csv")
    
    if os.path.exists(data_file):
        print(f"\n✓ Found existing weather data at {data_file}")
        try:
            df = fetcher.load_data()
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            print("  Fetching fresh data...")
            df = fetcher.fetch_and_save(start_date="1940-01-01")
    else:
        print("\n⟳ Fetching historical weather data (this may take a minute)...")
        try:
            df = fetcher.fetch_and_save(start_date="1940-01-01")
        except Exception as e:
            print(f"\n✗ Error fetching data: {e}")
            print("Please check your internet connection and try again.")
            return 1
    
    print(f"\n✓ Loaded {len(df):,} days of weather data")
    print(f"  Date range: {df['date'].min().strftime('%B %d, %Y')} to {df['date'].max().strftime('%B %d, %Y')}")
    
    # Analyze data
    print_section("Analyzing Weather Patterns")
    analyzer = WeatherAnalyzer(df)
    
    # Calculate seasonal features
    print("\n⟳ Calculating seasonal features...")
    seasonal_df = analyzer.calculate_seasonal_features()
    print(f"✓ Analyzed {len(seasonal_df)} years of seasonal data")
    
    # Calculate winter-specific features
    print("\n⟳ Analyzing historical winters...")
    winter_df = analyzer.calculate_winter_features()
    print(f"✓ Analyzed {len(winter_df)} complete winters")
    
    # Show recent winter summary
    print("\n  Recent Winters:")
    recent_winters = winter_df.tail(5)
    for _, winter in recent_winters.iterrows():
        print(f"    {winter['winter_label']}: {winter['severity_category']:10s} "
              f"(Avg: {format_temp(winter['avg_temp'])}, Snow: {format_precip(winter['total_snowfall'])})")
    
    # Analyze correlations
    print("\n⟳ Analyzing seasonal correlations...")
    correlation_results = analyzer.analyze_seasonal_correlations(seasonal_df)
    correlations = correlation_results["correlations"]
    correlation_df = correlation_results["correlation_df"]
    
    print("✓ Correlation analysis complete")
    print("\n  Key Findings:")
    
    if correlations.get("summer_temp_to_winter_severity"):
        corr = correlations["summer_temp_to_winter_severity"]
        direction = "warmer" if corr < 0 else "colder"
        print(f"    • Summer temperature vs Winter severity: {corr:.3f}")
        print(f"      → Hot summers tend to correlate with {direction} winters")
    
    if correlations.get("summer_temp_to_winter_snowfall"):
        corr = correlations["summer_temp_to_winter_snowfall"]
        direction = "more" if corr > 0 else "less"
        print(f"    • Summer temperature vs Winter snowfall: {corr:.3f}")
        print(f"      → Hot summers tend to see {direction} snowfall the following winter")
    
    if correlations.get("fall_temp_to_winter_temp"):
        corr = correlations["fall_temp_to_winter_temp"]
        print(f"    • Fall temperature vs Winter temperature: {corr:.3f}")
        if corr > 0.3:
            print(f"      → Warm falls tend to lead to milder winters")
        elif corr < -0.3:
            print(f"      → Warm falls tend to lead to colder winters")
    
    # Train prediction model
    print_section("Training Prediction Model")
    predictor = WinterPredictor()
    
    try:
        predictor.train(correlation_df)
        print("\n✓ Model training complete")
    except Exception as e:
        print(f"\n✗ Error training model: {e}")
        return 1
    
    # Make prediction for 2025-2026 winter
    print_section("Winter 2025-2026 Prediction")
    
    try:
        prediction = predictor.predict_from_current_data(analyzer, target_winter_year=2026)
        
        print(f"\nBased on weather patterns from {prediction['based_on_year']}:")
        print(f"\n  Predicted Category: {prediction['predicted_category']}")
        print(f"  Confidence: {prediction['confidence'] * 100:.1f}%")
        print(f"\n  Predicted Average Temperature: {format_temp(prediction['predicted_avg_temp'])}")
        print(f"  Predicted Total Snowfall: {format_precip(prediction['predicted_snowfall'])}")
        print(f"  Severity Score: {prediction['severity_score']:.1f}")
        
        print("\n  Category Probabilities:")
        for category, prob in sorted(prediction['category_probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
            bar_length = int(prob * 30)
            bar = "█" * bar_length
            print(f"    {category:10s} [{bar:30s}] {prob * 100:5.1f}%")
        
        # Historical comparison
        print("\n⟳ Comparing to historical winters...")
        comparison = predictor.get_historical_comparison(prediction, winter_df)
        
        print(f"\n  Historical Context:")
        print(f"    • This prediction ranks at the {comparison['percentile_rank']:.0f}th percentile for severity")
        print(f"    • Historical average: {format_temp(comparison['historical_avg_temp'])}, "
              f"{format_precip(comparison['historical_avg_snowfall'])} snow")
        
        print("\n  Similar Historical Winters:")
        for winter in comparison['similar_winters']:
            print(f"    • {winter['winter_label']}: {winter['severity_category']} "
                  f"({format_temp(winter['avg_temp'])}, {format_precip(winter['total_snowfall'])})")
        
    except Exception as e:
        print(f"\n✗ Error making prediction: {e}")
        print("\nNote: Predictions require complete seasonal data from the preceding year.")
        return 1
    
    # Recent seasonal summary
    print_section("Recent Seasonal Trends (Last 5 Years)")
    recent_summary = analyzer.get_recent_seasons_summary(n_years=5)
    
    for season, stats in recent_summary.items():
        print(f"\n  {season.capitalize()}:")
        print(f"    • Average Temperature: {format_temp(stats['avg_temp'])}")
        print(f"    • Average Annual Precipitation: {format_precip(stats['total_precip'])}")
        print(f"    • Trend: {stats['temp_trend'].capitalize()}")
    
    # Summary
    print_section("Summary")
    print("\nThis analysis examined historical weather patterns in Boise, Idaho to predict")
    print("the severity of Winter 2025-2026. The model uses machine learning to identify")
    print("relationships between seasonal weather patterns (summer and fall conditions)")
    print("and subsequent winter characteristics.")
    print("\nKey factors influencing winter predictions include:")
    print("  • Summer temperatures and extreme heat days")
    print("  • Fall temperature patterns and precipitation")
    print("  • Historical correlation patterns between seasons")
    print("\nThe prediction model can be updated and rerun as more data becomes available")
    print("throughout 2025 to refine the winter 2025-2026 forecast.")
    
    print_section("Application Complete")
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
