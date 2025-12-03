"""
Visualization module for winter predictions and historical data.
Creates charts and graphs for weather analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os


class WeatherVisualizer:
    """Creates visualizations for weather data and predictions."""
    
    # ENSO phase abbreviations for compact display
    ENSO_PHASE_ABBREVIATIONS = {
        'Neutral': 'N',
        'El Niño': 'EN',
        'La Niña': 'LN'
    }
    
    def __init__(self, output_dir="visualizations"):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set the style for better-looking plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def visualize_prediction(self, prediction, winter_df=None, enso_info=None):
        """Create visualization for winter prediction.
        
        Args:
            prediction: Dictionary with prediction results
            winter_df: Optional DataFrame with historical winter data
            enso_info: Optional dict with ENSO classification for the predicted winter
            
        Returns:
            str: Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Add ENSO to title if available
        title = f"Winter {prediction['winter_label']} Prediction (Dec-Jan-Feb)"
        if enso_info and enso_info.get('phase') != 'Unknown':
            title += f"\n{enso_info['description']}"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Category Probabilities (top-left)
        ax1 = axes[0, 0]
        categories = list(prediction['category_probabilities'].keys())
        probabilities = [prediction['category_probabilities'][cat] * 100 
                        for cat in categories]
        
        colors = ['#4CAF50', '#FFC107', '#FF5722', '#9C27B0'][:len(categories)]
        bars = ax1.barh(categories, probabilities, color=colors, alpha=0.7)
        
        # Highlight the predicted category
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
        
        # Add ENSO information if available
        if enso_info and enso_info.get('phase') != 'Unknown':
            feature_text += f"ENSO Phase: {enso_info['phase']}\n"
            if enso_info.get('strength') and enso_info['strength'] != 'N/A':
                feature_text += f"Strength: {enso_info['strength']}\n"
            feature_text += f"ONI: {enso_info['oni_value']:+.1f}°C\n\n"
        
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
        
        # Create figure with custom GridSpec layout
        # Top row: 2 equal columns
        # Bottom row: 1 column left, 2 smaller columns right for pie chart and ENSO table
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 0.5, 0.5], height_ratios=[1, 1])
        
        fig.suptitle(f'Last {n_years} Winters in Boise (Historical Data)\nWinter = December-January-February', 
                     fontsize=16, fontweight='bold')
        
        # 1. Temperature Trends (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        
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
        
        # 2. Snowfall Trends (top-right, spans 2 columns)
        ax2 = fig.add_subplot(gs[0, 1:])
        
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
        ax3 = fig.add_subplot(gs[1, 0])
        
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
        
        # 4. Severity Distribution Pie Chart (bottom-center-right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Create pie chart showing severity distribution
        category_counts = recent_winters['severity_category'].value_counts()
        colors_pie = [severity_colors.get(cat, '#64B5F6') for cat in category_counts.index]
        
        wedges, texts, autotexts = ax4.pie(category_counts.values, 
                                           labels=category_counts.index,
                                           colors=colors_pie,
                                           autopct='%1.0f%%',
                                           startangle=90,
                                           textprops={'fontsize': 9, 'fontweight': 'bold'})
        
        ax4.set_title(f'Severity Distribution\n(Last {n_years} Years)', fontweight='bold', fontsize=10)
        
        # 5. ENSO Phase Table (bottom-far-right)
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Check if ENSO data is available
        if 'enso_phase' in recent_winters.columns:
            # Create a table showing ENSO phases for each winter
            table_data = []
            for _, row in recent_winters.iterrows():
                enso_short = self.ENSO_PHASE_ABBREVIATIONS.get(row['enso_phase'], '?')
                table_data.append([row['winter_label'], enso_short])
            
            # Create text display
            separator_width = 15
            enso_text = "ENSO Phase\n\n"
            enso_text += "Year     ENSO\n"
            enso_text += "─" * separator_width + "\n"
            for row_data in table_data:
                enso_text += f"{row_data[0]:<9}{row_data[1]}\n"
            
            enso_text += "\n" + "─" * separator_width + "\n"
            enso_text += "EN = El Niño\n"
            enso_text += "LN = La Niña\n"
            enso_text += "N  = Neutral"
            
            ax5.text(0.1, 0.95, enso_text, transform=ax5.transAxes,
                    fontsize=8, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            ax5.axis('off')
            ax5.set_title('ENSO Phases', fontweight='bold', fontsize=10)
        else:
            # If no ENSO data, just turn off the axis
            ax5.axis('off')
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"historical_winters_last_{n_years}_years.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def visualize_both(self, prediction, winter_df, n_years=10, enso_info=None):
        """Create both visualizations and return their paths.
        
        Args:
            prediction: Dictionary with prediction results
            winter_df: DataFrame with historical winter data
            n_years: Number of recent years to visualize
            enso_info: Optional dict with ENSO classification for the predicted winter
            
        Returns:
            tuple: (prediction_path, historical_path)
        """
        prediction_path = self.visualize_prediction(prediction, winter_df, enso_info)
        historical_path = self.visualize_historical_winters(winter_df, n_years)
        
        return prediction_path, historical_path
    
    def visualize_feature_analysis(self, correlation_df, prediction=None):
        """Create visualization showing all features used in the analysis.
        
        Args:
            correlation_df: DataFrame with correlation data including all features
            prediction: Optional prediction dict to highlight current prediction features
            
        Returns:
            str: Path to saved figure
        """
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.35)
        
        fig.suptitle('Winter Prediction Feature Analysis\nComprehensive View of All Input Features', 
                     fontsize=16, fontweight='bold')
        
        # 1. Feature Correlation Heatmap (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Select numeric columns for correlation (exclude categorical columns)
        feature_cols = [col for col in correlation_df.columns 
                       if (col.startswith('prev_') or col.startswith('enso_') 
                           or col.startswith('rolling_') or col.startswith('winter_'))
                       and col not in ['winter_category', 'winter_label', 'winter_year']
                       and pd.api.types.is_numeric_dtype(correlation_df[col])]
        
        if len(feature_cols) > 0:
            corr_matrix = correlation_df[feature_cols].corr()
            
            # Create a mask for the upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Plot heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                       center=0, vmin=-1, vmax=1, ax=ax1, cbar_kws={'label': 'Correlation'})
            ax1.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=11, pad=10)
            
            # Rotate labels for better readability
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=7)
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=7)
        
        # 2. Feature Categories Breakdown (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Count features by category
        feature_categories = {
            'Summer Features': len([c for c in correlation_df.columns if 'summer' in c and c.startswith('prev_')]),
            'Fall Features': len([c for c in correlation_df.columns if 'fall' in c and c.startswith('prev_')]),
            'ENSO Features': len([c for c in correlation_df.columns if c.startswith('enso_')]),
            'Previous Winter': len([c for c in correlation_df.columns if 'prev_winter' in c]),
            'Rolling Averages': len([c for c in correlation_df.columns if c.startswith('rolling_')])
        }
        
        categories = list(feature_categories.keys())
        counts = list(feature_categories.values())
        colors_cat = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        wedges, texts, autotexts = ax2.pie(counts, labels=categories, colors=colors_cat,
                                            autopct='%1.0f%%', startangle=90,
                                            textprops={'fontsize': 7, 'fontweight': 'bold'})
        ax2.set_title('Feature Categories', fontweight='bold', fontsize=11, pad=10)
        
        # 3. Lag Features Over Time (middle row, left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        if 'winter_year' in correlation_df.columns and 'prev_winter_severity' in correlation_df.columns:
            recent_data = correlation_df.tail(15)
            
            ax3.plot(recent_data['winter_year'], recent_data['prev_winter_severity'], 
                    marker='o', label='Prev Winter Severity', linewidth=2, markersize=6)
            
            if 'rolling_2yr_severity' in recent_data.columns:
                ax3.plot(recent_data['winter_year'], recent_data['rolling_2yr_severity'],
                        marker='s', label='2-Year Avg', linewidth=2, markersize=5, linestyle='--', alpha=0.7)
            
            if 'rolling_3yr_severity' in recent_data.columns:
                ax3.plot(recent_data['winter_year'], recent_data['rolling_3yr_severity'],
                        marker='^', label='3-Year Avg', linewidth=2, markersize=5, linestyle=':', alpha=0.7)
            
            ax3.set_xlabel('Winter Year', fontweight='bold', fontsize=9)
            ax3.set_ylabel('Severity Score', fontweight='bold', fontsize=9)
            ax3.set_title('Lag Features: Winter Severity Trends', fontweight='bold', fontsize=10, pad=10)
            ax3.legend(fontsize=7, loc='upper left')
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
        else:
            ax3.text(0.5, 0.5, 'Lag feature data not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.axis('off')
        
        # 4. Snowfall Lag Features (middle row, center)
        ax4 = fig.add_subplot(gs[1, 1])
        
        if 'winter_year' in correlation_df.columns and 'prev_winter_snowfall' in correlation_df.columns:
            recent_data = correlation_df.tail(15)
            
            ax4.plot(recent_data['winter_year'], recent_data['prev_winter_snowfall'],
                    marker='o', label='Prev Winter Snowfall', linewidth=2, markersize=6, color='#2196F3')
            
            if 'rolling_2yr_snowfall' in recent_data.columns:
                ax4.plot(recent_data['winter_year'], recent_data['rolling_2yr_snowfall'],
                        marker='s', label='2-Year Avg', linewidth=2, markersize=5, 
                        linestyle='--', alpha=0.7, color='#64B5F6')
            
            if 'rolling_3yr_snowfall' in recent_data.columns:
                ax4.plot(recent_data['winter_year'], recent_data['rolling_3yr_snowfall'],
                        marker='^', label='3-Year Avg', linewidth=2, markersize=5,
                        linestyle=':', alpha=0.7, color='#90CAF9')
            
            ax4.set_xlabel('Winter Year', fontweight='bold', fontsize=9)
            ax4.set_ylabel('Snowfall (inches)', fontweight='bold', fontsize=9)
            ax4.set_title('Lag Features: Snowfall Trends', fontweight='bold', fontsize=10, pad=10)
            ax4.legend(fontsize=7, loc='upper left')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
        else:
            ax4.text(0.5, 0.5, 'Snowfall lag data not available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.axis('off')
        
        # 5. Temperature Lag Features (middle row, right)
        ax5 = fig.add_subplot(gs[1, 2])
        
        if 'winter_year' in correlation_df.columns and 'prev_winter_temp_avg' in correlation_df.columns:
            recent_data = correlation_df.tail(15)
            
            ax5.plot(recent_data['winter_year'], recent_data['prev_winter_temp_avg'],
                    marker='o', label='Prev Winter Temp', linewidth=2, markersize=6, color='#FF5722')
            
            if 'rolling_2yr_temp' in recent_data.columns:
                ax5.plot(recent_data['winter_year'], recent_data['rolling_2yr_temp'],
                        marker='s', label='2-Year Avg', linewidth=2, markersize=5,
                        linestyle='--', alpha=0.7, color='#FF8A65')
            
            if 'rolling_3yr_temp' in recent_data.columns:
                ax5.plot(recent_data['winter_year'], recent_data['rolling_3yr_temp'],
                        marker='^', label='3-Year Avg', linewidth=2, markersize=5,
                        linestyle=':', alpha=0.7, color='#FFAB91')
            
            ax5.set_xlabel('Winter Year', fontweight='bold', fontsize=9)
            ax5.set_ylabel('Temperature (°F)', fontweight='bold', fontsize=9)
            ax5.set_title('Lag Features: Temperature Trends', fontweight='bold', fontsize=10, pad=10)
            ax5.legend(fontsize=7, loc='upper left')
            ax5.grid(True, alpha=0.3)
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
        else:
            ax5.text(0.5, 0.5, 'Temperature lag data not available',
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.axis('off')
        
        # 6. Feature List (bottom row, left and center columns)
        ax6 = fig.add_subplot(gs[2, :2])
        
        # Create a comprehensive feature list
        feature_list_text = "Complete Feature Set Used for Prediction:\n\n"
        
        # Organize features by category
        summer_features = sorted([c for c in correlation_df.columns if 'summer' in c and c.startswith('prev_')])
        fall_features = sorted([c for c in correlation_df.columns if 'fall' in c and c.startswith('prev_')])
        enso_features = sorted([c for c in correlation_df.columns if c.startswith('enso_')])
        lag_features = sorted([c for c in correlation_df.columns if 'prev_winter' in c])
        rolling_features = sorted([c for c in correlation_df.columns if c.startswith('rolling_')])
        
        if summer_features:
            feature_list_text += "Summer (Prev Year):\n"
            for feat in summer_features:
                feature_list_text += f"  • {feat}\n"
        
        if fall_features:
            feature_list_text += "\nFall (Prev Year):\n"
            for feat in fall_features:
                feature_list_text += f"  • {feat}\n"
        
        if enso_features:
            feature_list_text += "\nENSO Indicators:\n"
            for feat in enso_features:
                feature_list_text += f"  • {feat}\n"
        
        if lag_features:
            feature_list_text += "\nPrevious Winter:\n"
            for feat in lag_features:
                feature_list_text += f"  • {feat}\n"
        
        if rolling_features:
            feature_list_text += "\nRolling Averages:\n"
            for feat in rolling_features:
                feature_list_text += f"  • {feat}\n"
        
        ax6.text(0.05, 0.90, feature_list_text, transform=ax6.transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        ax6.axis('off')
        ax6.set_title('Feature Inventory', fontweight='bold', fontsize=11, loc='left', pad=10)
        
        # 7. Key Statistics (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        
        stats_text = "Dataset Statistics:\n\n"
        stats_text += f"Total Records: {len(correlation_df)}\n\n"
        
        if 'winter_severity' in correlation_df.columns:
            stats_text += f"Severity Range:\n"
            stats_text += f"  Min: {correlation_df['winter_severity'].min():.1f}\n"
            stats_text += f"  Max: {correlation_df['winter_severity'].max():.1f}\n"
            stats_text += f"  Mean: {correlation_df['winter_severity'].mean():.1f}\n\n"
        
        if 'winter_snowfall' in correlation_df.columns:
            stats_text += f"Snowfall Range:\n"
            stats_text += f"  Min: {correlation_df['winter_snowfall'].min():.1f}\"\n"
            stats_text += f"  Max: {correlation_df['winter_snowfall'].max():.1f}\"\n"
            stats_text += f"  Mean: {correlation_df['winter_snowfall'].mean():.1f}\"\n\n"
        
        # Add feature count
        total_features = len([c for c in correlation_df.columns 
                             if c.startswith('prev_') or c.startswith('enso_') 
                             or c.startswith('rolling_')])
        stats_text += f"Total Features: {total_features}\n"
        
        # If prediction is provided, add current values
        if prediction and 'input_features' in prediction:
            stats_text += f"\nCurrent Prediction:\n"
            stats_text += f"  Category: {prediction['predicted_category']}\n"
            stats_text += f"  Confidence: {prediction['confidence']*100:.1f}%\n"
        
        ax7.text(0.1, 0.90, stats_text, transform=ax7.transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax7.axis('off')
        ax7.set_title('Summary Statistics', fontweight='bold', fontsize=11, loc='left', pad=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save the figure
        filename = "feature_analysis.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
