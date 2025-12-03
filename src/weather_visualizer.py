"""
Visualization module for winter predictions and historical data.
Creates charts and graphs for weather analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os


class WeatherVisualizer:
    """Creates visualizations for weather data and predictions."""
    
    def __init__(self, output_dir="visualizations"):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist (thread-safe)
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the style for better-looking plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def visualize_prediction(self, prediction, winter_df=None):
        """Create visualization for winter prediction.
        
        Args:
            prediction: Dictionary with prediction results
            winter_df: Optional DataFrame with historical winter data
            
        Returns:
            str: Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Winter {prediction['winter_label']} Prediction", 
                     fontsize=16, fontweight='bold')
        
        # 1. Category Probabilities (top-left)
        ax1 = axes[0, 0]
        categories = list(prediction['category_probabilities'].keys())
        probabilities = [prediction['category_probabilities'][cat] * 100 
                        for cat in categories]
        
        colors = ['#4CAF50', '#FFC107', '#FF5722', '#9C27B0'][:len(categories)]
        bars = ax1.barh(categories, probabilities, color=colors, alpha=0.7)
        
        # Highlight the predicted category (with validation)
        if prediction['predicted_category'] in categories:
            predicted_idx = categories.index(prediction['predicted_category'])
            bars[predicted_idx].set_alpha(1.0)
            bars[predicted_idx].set_edgecolor('black')
            bars[predicted_idx].set_linewidth(2)
        
        ax1.set_xlabel('Probability (%)', fontweight='bold')
        ax1.set_title('Winter Severity Probabilities', fontweight='bold')
        ax1.set_xlim(0, 105)
        
        # Add percentage labels
        for i, v in enumerate(probabilities):
            ax1.text(v + 2, i, f'{v:.1f}%', va='center')
        
        # 2. Temperature and Snowfall Prediction (top-right)
        ax2 = axes[0, 1]
        
        metrics = ['Avg Temp\n(°F)', 'Total Snowfall\n(inches)']
        values = [prediction['predicted_avg_temp'], prediction['predicted_snowfall']]
        
        if winter_df is not None:
            historical_avg_temp = winter_df['avg_temp'].mean()
            historical_avg_snow = winter_df['total_snowfall'].mean()
            historical_values = [historical_avg_temp, historical_avg_snow]
        else:
            historical_values = [0, 0]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, values, width, label='Predicted', 
                       color='#2196F3', alpha=0.8)
        if winter_df is not None:
            bars2 = ax2.bar(x + width/2, historical_values, width, 
                          label='Historical Avg', color='#9E9E9E', alpha=0.6)
        
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.set_title('Predicted vs Historical Average', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        if winter_df is not None:
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Severity Score Gauge (bottom-left)
        ax3 = axes[1, 0]
        severity_score = prediction['severity_score']
        
        # Create a gauge-like visualization
        categories_thresholds = [
            (0, 15, 'Mild', '#4CAF50'),
            (15, 30, 'Moderate', '#FFC107'),
            (30, 50, 'Severe', '#FF5722'),
            (50, 100, 'Extreme', '#9C27B0')
        ]
        
        # Draw the gauge
        for start, end, cat_name, color in categories_thresholds:
            ax3.barh(0, end - start, left=start, height=0.5, 
                    color=color, alpha=0.3, edgecolor='black', linewidth=0.5)
            # Add category label
            mid = (start + end) / 2
            ax3.text(mid, 0, cat_name, ha='center', va='center', 
                    fontsize=9, fontweight='bold')
        
        # Add the severity score indicator
        ax3.plot([severity_score, severity_score], [-0.3, 0.3], 
                color='red', linewidth=3, marker='v', markersize=10, label='Predicted')
        
        ax3.set_xlim(0, 70)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_xlabel('Severity Score', fontweight='bold')
        ax3.set_title(f'Winter Severity: {prediction["predicted_category"]}', fontweight='bold')
        ax3.set_yticks([])
        ax3.legend(loc='upper right')
        
        # 4. Input Features (bottom-right)
        ax4 = axes[1, 1]
        
        # Display the features used for prediction
        feature_text = f"Prediction Based on {prediction['based_on_year']} Data\n\n"
        
        if 'input_features' in prediction:
            feature_text += "Summer Features:\n"
            for key, value in prediction['input_features']['summer'].items():
                feature_text += f"  • {key}: {value:.2f}\n"
            
            feature_text += "\nFall Features:\n"
            for key, value in prediction['input_features']['fall'].items():
                feature_text += f"  • {key}: {value:.2f}\n"
        
        feature_text += f"\nConfidence: {prediction['confidence'] * 100:.1f}%"
        
        ax4.text(0.1, 0.9, feature_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax4.axis('off')
        ax4.set_title('Prediction Inputs', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"winter_{prediction['winter_label']}_prediction.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def visualize_historical_winters(self, winter_df, n_years=10):
        """Create visualization for historical winter data.
        
        Args:
            winter_df: DataFrame with historical winter data
            n_years: Number of recent years to visualize
            
        Returns:
            str: Path to saved figure
        """
        # Get the last n years
        recent_winters = winter_df.tail(n_years)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Last {n_years} Winters in Boise (Historical Data)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Temperature Trends (top-left)
        ax1 = axes[0, 0]
        
        ax1.plot(recent_winters['winter_label'], recent_winters['avg_temp'], 
                marker='o', linewidth=2, markersize=8, color='#2196F3', label='Average Temp')
        ax1.plot(recent_winters['winter_label'], recent_winters['avg_max_temp'], 
                marker='s', linewidth=1, markersize=6, color='#FF5722', 
                alpha=0.6, linestyle='--', label='Avg Max Temp')
        ax1.plot(recent_winters['winter_label'], recent_winters['avg_min_temp'], 
                marker='s', linewidth=1, markersize=6, color='#4CAF50', 
                alpha=0.6, linestyle='--', label='Avg Min Temp')
        
        ax1.set_xlabel('Winter', fontweight='bold')
        ax1.set_ylabel('Temperature (°F)', fontweight='bold')
        ax1.set_title('Winter Temperature Trends', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Snowfall Trends (top-right)
        ax2 = axes[0, 1]
        
        bars = ax2.bar(recent_winters['winter_label'], recent_winters['total_snowfall'], 
                      color='#64B5F6', edgecolor='black', alpha=0.7)
        
        # Color bars by severity
        severity_colors = {
            'Mild': '#4CAF50',
            'Moderate': '#FFC107',
            'Severe': '#FF5722',
            'Extreme': '#9C27B0'
        }
        
        for i, (idx, row) in enumerate(recent_winters.iterrows()):
            bars[i].set_color(severity_colors.get(row['severity_category'], '#64B5F6'))
        
        ax2.set_xlabel('Winter', fontweight='bold')
        ax2.set_ylabel('Total Snowfall (inches)', fontweight='bold')
        ax2.set_title('Winter Snowfall by Year', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}"', ha='center', va='bottom', fontsize=8)
        
        # 3. Severity Categories (bottom-left)
        ax3 = axes[1, 0]
        
        # Create scatter plot with severity score
        severities = recent_winters['severity_category'].map({
            'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Extreme': 4
        })
        
        colors_list = [severity_colors.get(cat, '#64B5F6') 
                      for cat in recent_winters['severity_category']]
        
        scatter = ax3.scatter(range(len(recent_winters)), 
                            recent_winters['severity_score'],
                            c=colors_list, s=200, alpha=0.6, edgecolors='black', linewidth=2)
        
        ax3.set_xticks(range(len(recent_winters)))
        ax3.set_xticklabels(recent_winters['winter_label'], rotation=45, ha='right')
        ax3.set_xlabel('Winter', fontweight='bold')
        ax3.set_ylabel('Severity Score', fontweight='bold')
        ax3.set_title('Winter Severity Scores', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add severity threshold lines
        ax3.axhline(y=15, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax3.axhline(y=30, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add labels for thresholds
        ax3.text(len(recent_winters) - 0.5, 15, 'Moderate', 
                ha='right', va='bottom', fontsize=8, alpha=0.7)
        ax3.text(len(recent_winters) - 0.5, 30, 'Severe', 
                ha='right', va='bottom', fontsize=8, alpha=0.7)
        ax3.text(len(recent_winters) - 0.5, 50, 'Extreme', 
                ha='right', va='bottom', fontsize=8, alpha=0.7)
        
        # 4. Category Distribution (bottom-right)
        ax4 = axes[1, 1]
        
        category_counts = recent_winters['severity_category'].value_counts()
        colors_pie = [severity_colors.get(cat, '#64B5F6') for cat in category_counts.index]
        
        wedges, texts, autotexts = ax4.pie(category_counts.values, 
                                           labels=category_counts.index,
                                           colors=colors_pie,
                                           autopct='%1.0f%%',
                                           startangle=90,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        ax4.set_title(f'Winter Severity Distribution\n(Last {n_years} Years)', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"historical_winters_last_{n_years}_years.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def visualize_both(self, prediction, winter_df, n_years=10):
        """Create both visualizations and return their paths.
        
        Args:
            prediction: Dictionary with prediction results
            winter_df: DataFrame with historical winter data
            n_years: Number of recent years to visualize
            
        Returns:
            tuple: (prediction_path, historical_path)
        """
        prediction_path = self.visualize_prediction(prediction, winter_df)
        historical_path = self.visualize_historical_winters(winter_df, n_years)
        
        return prediction_path, historical_path
