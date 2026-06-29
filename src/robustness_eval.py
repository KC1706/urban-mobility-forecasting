#!/usr/bin/env python3
"""
Robustness Evaluation Framework for Urban Mobility Models

Evaluates model performance across multiple dimensions:
- Spatial robustness: Performance across different regions (boroughs, neighborhoods)
- Temporal robustness: Performance across different time periods (hours, days, seasons)
- Stability analysis: Variance in predictions over time
- Edge case performance: Behavior during extreme demand events

This addresses the professor's requirement for robustness analysis beyond overall accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import scipy.stats as stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustnessEvaluator:
    """
    Comprehensive robustness evaluation for urban mobility models
    """
    
    def __init__(self, task_type: str = "regression"):
        """
        Initialize robustness evaluator
        
        Args:
            task_type: "regression" for demand prediction, "classification" for categories
        """
        self.task_type = task_type
        self.results = {}
        self.visualization_data = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics based on task type"""
        
        if self.task_type == "regression":
            return {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            }
        else:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred, average='weighted'),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted')
            }
    
    def spatial_robustness_analysis(self, data: pd.DataFrame, predictions: np.ndarray, 
                                  spatial_column: str = 'pickup_borough') -> Dict[str, Any]:
        """
        Evaluate model performance across different spatial regions
        
        Args:
            data: DataFrame with spatial information and actual values
            predictions: Model predictions
            spatial_column: Column name containing spatial identifiers
            
        Returns:
            Dictionary with spatial robustness metrics
        """
        logger.info("Performing spatial robustness analysis...")
        
        results = {}
        target_col = 'trip_count' if 'trip_count' in data.columns else 'demand'
        
        # Get unique spatial regions
        regions = data[spatial_column].unique()
        
        region_metrics = {}
        region_counts = {}
        
        for region in regions:
            # Filter data for this region
            mask = data[spatial_column] == region
            if mask.sum() == 0:
                continue
                
            region_actual = data[mask][target_col].values
            region_pred = predictions[mask]
            
            # Calculate metrics for this region
            metrics = self.calculate_metrics(region_actual, region_pred)
            region_metrics[region] = metrics
            region_counts[region] = len(region_actual)
        
        # Calculate overall statistics
        primary_metric = 'rmse' if self.task_type == 'regression' else 'f1'
        metric_values = [metrics[primary_metric] for metrics in region_metrics.values()]
        
        results = {
            'region_metrics': region_metrics,
            'region_counts': region_counts,
            'metric_statistics': {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values),
                'min': np.min(metric_values),
                'max': np.max(metric_values),
                'cv': np.std(metric_values) / np.mean(metric_values)  # Coefficient of variation
            },
            'spatial_column': spatial_column,
            'primary_metric': primary_metric
        }
        
        self.results['spatial_robustness'] = results
        logger.info(f"Spatial analysis complete: {len(regions)} regions analyzed")
        
        return results
    
    def temporal_robustness_analysis(self, data: pd.DataFrame, predictions: np.ndarray,
                                   datetime_column: str = 'pickup_datetime') -> Dict[str, Any]:
        """
        Evaluate model performance across different temporal periods
        
        Args:
            data: DataFrame with datetime information and actual values  
            predictions: Model predictions
            datetime_column: Column name containing datetime information
            
        Returns:
            Dictionary with temporal robustness metrics
        """
        logger.info("Performing temporal robustness analysis...")
        
        # Ensure datetime column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(data[datetime_column]):
            data[datetime_column] = pd.to_datetime(data[datetime_column])
        
        target_col = 'trip_count' if 'trip_count' in data.columns else 'demand'
        results = {}
        
        # Extract temporal features
        data_temp = data.copy()
        data_temp['hour'] = data_temp[datetime_column].dt.hour
        data_temp['day_of_week'] = data_temp[datetime_column].dt.dayofweek
        data_temp['month'] = data_temp[datetime_column].dt.month
        data_temp['is_weekend'] = data_temp['day_of_week'].isin([5, 6])
        
        # 1. Hourly performance
        hourly_metrics = {}
        for hour in range(24):
            mask = data_temp['hour'] == hour
            if mask.sum() > 0:
                hour_actual = data_temp[mask][target_col].values
                hour_pred = predictions[mask]
                hourly_metrics[hour] = self.calculate_metrics(hour_actual, hour_pred)
        
        # 2. Day of week performance
        dow_metrics = {}
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for dow in range(7):
            mask = data_temp['day_of_week'] == dow
            if mask.sum() > 0:
                dow_actual = data_temp[mask][target_col].values
                dow_pred = predictions[mask]
                dow_metrics[dow_names[dow]] = self.calculate_metrics(dow_actual, dow_pred)
        
        # 3. Weekend vs Weekday
        weekend_weekday_metrics = {}
        for is_weekend in [True, False]:
            mask = data_temp['is_weekend'] == is_weekend
            if mask.sum() > 0:
                period_actual = data_temp[mask][target_col].values
                period_pred = predictions[mask]
                period_name = 'Weekend' if is_weekend else 'Weekday'
                weekend_weekday_metrics[period_name] = self.calculate_metrics(period_actual, period_pred)
        
        # 4. Monthly performance (seasonal trends)
        monthly_metrics = {}
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in range(1, 13):
            mask = data_temp['month'] == month
            if mask.sum() > 0:
                month_actual = data_temp[mask][target_col].values
                month_pred = predictions[mask]
                monthly_metrics[month_names[month-1]] = self.calculate_metrics(month_actual, month_pred)
        
        # Calculate temporal variance statistics
        primary_metric = 'rmse' if self.task_type == 'regression' else 'f1'
        
        hourly_variance = np.var([metrics[primary_metric] for metrics in hourly_metrics.values()])
        dow_variance = np.var([metrics[primary_metric] for metrics in dow_metrics.values()])
        monthly_variance = np.var([metrics[primary_metric] for metrics in monthly_metrics.values()])
        
        results = {
            'hourly_metrics': hourly_metrics,
            'day_of_week_metrics': dow_metrics,
            'weekend_weekday_metrics': weekend_weekday_metrics,
            'monthly_metrics': monthly_metrics,
            'temporal_variance': {
                'hourly_variance': hourly_variance,
                'dow_variance': dow_variance,
                'monthly_variance': monthly_variance
            },
            'primary_metric': primary_metric
        }
        
        self.results['temporal_robustness'] = results
        logger.info("Temporal robustness analysis complete")
        
        return results
    
    def stability_analysis(self, data: pd.DataFrame, predictions: np.ndarray,
                          datetime_column: str = 'pickup_datetime', 
                          window_size: str = 'D') -> Dict[str, Any]:
        """
        Analyze prediction stability over time
        
        Args:
            data: DataFrame with datetime and actual values
            predictions: Model predictions
            datetime_column: Column name containing datetime
            window_size: Time window for aggregation ('H', 'D', 'W')
            
        Returns:
            Dictionary with stability metrics
        """
        logger.info("Performing stability analysis...")
        
        # Prepare data
        if not pd.api.types.is_datetime64_any_dtype(data[datetime_column]):
            data[datetime_column] = pd.to_datetime(data[datetime_column])
        
        target_col = 'trip_count' if 'trip_count' in data.columns else 'demand'
        
        # Create DataFrame for analysis
        stability_data = pd.DataFrame({
            'datetime': data[datetime_column],
            'actual': data[target_col],
            'predicted': predictions
        })
        
        # Aggregate by time window
        stability_agg = stability_data.set_index('datetime').resample(window_size).agg({
            'actual': 'sum',
            'predicted': 'sum'
        }).reset_index()
        
        # Calculate rolling metrics
        window_periods = min(7, len(stability_agg) // 2)  # 7-period window or half the data
        
        rolling_metrics = []
        primary_metric = 'rmse' if self.task_type == 'regression' else 'f1'
        
        for i in range(window_periods, len(stability_agg)):
            window_data = stability_agg.iloc[i-window_periods:i]
            metrics = self.calculate_metrics(window_data['actual'].values, 
                                           window_data['predicted'].values)
            metrics['datetime'] = stability_agg.iloc[i]['datetime']
            rolling_metrics.append(metrics)
        
        rolling_df = pd.DataFrame(rolling_metrics)
        
        # Calculate stability statistics
        stability_stats = {
            'mean_performance': rolling_df[primary_metric].mean(),
            'std_performance': rolling_df[primary_metric].std(),
            'cv_performance': rolling_df[primary_metric].std() / rolling_df[primary_metric].mean(),
            'trend_slope': self._calculate_trend_slope(rolling_df, primary_metric),
            'num_windows': len(rolling_df)
        }
        
        results = {
            'rolling_metrics': rolling_df,
            'stability_statistics': stability_stats,
            'aggregated_data': stability_agg,
            'window_size': window_size,
            'primary_metric': primary_metric
        }
        
        self.results['stability_analysis'] = results
        logger.info(f"Stability analysis complete: {len(rolling_df)} time windows analyzed")
        
        return results
    
    def _calculate_trend_slope(self, df: pd.DataFrame, metric_col: str) -> float:
        """Calculate trend slope for a metric over time"""
        if len(df) < 2:
            return 0.0
        
        x = np.arange(len(df))
        y = df[metric_col].values
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    def extreme_events_analysis(self, data: pd.DataFrame, predictions: np.ndarray,
                               target_col: str = None, percentile_thresholds: Tuple[float, float] = (5, 95)) -> Dict[str, Any]:
        """
        Analyze model performance during extreme demand events
        
        Args:
            data: DataFrame with actual values
            predictions: Model predictions  
            target_col: Target column name
            percentile_thresholds: (low, high) percentiles for defining extreme events
            
        Returns:
            Dictionary with extreme events analysis
        """
        logger.info("Performing extreme events analysis...")
        
        if target_col is None:
            target_col = 'trip_count' if 'trip_count' in data.columns else 'demand'
        
        actual_values = data[target_col].values
        
        # Define extreme events based on percentiles
        low_threshold = np.percentile(actual_values, percentile_thresholds[0])
        high_threshold = np.percentile(actual_values, percentile_thresholds[1])
        
        # Categorize events
        low_extreme_mask = actual_values <= low_threshold
        high_extreme_mask = actual_values >= high_threshold
        normal_mask = ~(low_extreme_mask | high_extreme_mask)
        
        # Calculate metrics for each category
        results = {}
        
        categories = {
            'low_extreme': low_extreme_mask,
            'normal': normal_mask, 
            'high_extreme': high_extreme_mask
        }
        
        for category, mask in categories.items():
            if mask.sum() > 0:
                cat_actual = actual_values[mask]
                cat_pred = predictions[mask]
                results[category] = {
                    'metrics': self.calculate_metrics(cat_actual, cat_pred),
                    'count': mask.sum(),
                    'percentage': (mask.sum() / len(actual_values)) * 100
                }
        
        # Calculate relative performance degradation
        normal_performance = results['normal']['metrics']
        primary_metric = 'rmse' if self.task_type == 'regression' else 'f1'
        
        performance_degradation = {}
        for category in ['low_extreme', 'high_extreme']:
            if category in results:
                cat_performance = results[category]['metrics'][primary_metric]
                normal_perf = normal_performance[primary_metric]
                
                if self.task_type == 'regression':
                    # For regression, higher RMSE/MAE is worse
                    degradation = ((cat_performance - normal_perf) / normal_perf) * 100
                else:
                    # For classification, lower F1/accuracy is worse  
                    degradation = ((normal_perf - cat_performance) / normal_perf) * 100
                
                performance_degradation[category] = degradation
        
        results['performance_degradation'] = performance_degradation
        results['thresholds'] = {
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'percentiles': percentile_thresholds
        }
        results['primary_metric'] = primary_metric
        
        self.results['extreme_events'] = results
        logger.info("Extreme events analysis complete")
        
        return results
    
    def generate_robustness_report(self, output_dir: str = "results/robustness") -> Dict[str, str]:
        """
        Generate comprehensive robustness analysis report
        
        Args:
            output_dir: Directory to save report and visualizations
            
        Returns:
            Dictionary with paths to generated files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = []
        generated_files = {}
        
        # Header
        report.append("# Model Robustness Analysis Report\n")
        report.append(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Task Type:** {self.task_type.title()}\n\n")
        
        # Spatial Robustness Section
        if 'spatial_robustness' in self.results:
            report.append("## 🗺️ Spatial Robustness Analysis\n")
            spatial_results = self.results['spatial_robustness']
            
            report.append("### Performance by Region\n")
            
            # Create spatial performance table
            spatial_df = pd.DataFrame(spatial_results['region_metrics']).T
            spatial_df['sample_count'] = [spatial_results['region_counts'][region] for region in spatial_df.index]
            
            report.append(spatial_df.round(4).to_markdown())
            report.append("\n\n")
            
            # Spatial statistics
            stats = spatial_results['metric_statistics']
            report.append("### Spatial Performance Statistics\n")
            report.append(f"- **Mean {spatial_results['primary_metric'].upper()}:** {stats['mean']:.4f}\n")
            report.append(f"- **Standard Deviation:** {stats['std']:.4f}\n")
            report.append(f"- **Coefficient of Variation:** {stats['cv']:.4f}\n")
            report.append(f"- **Min Performance:** {stats['min']:.4f}\n")
            report.append(f"- **Max Performance:** {stats['max']:.4f}\n\n")
            
            # Generate spatial visualization
            spatial_viz_path = output_path / "spatial_performance.png"
            self._create_spatial_visualization(spatial_results, spatial_viz_path)
            generated_files['spatial_visualization'] = str(spatial_viz_path)
        
        # Temporal Robustness Section  
        if 'temporal_robustness' in self.results:
            report.append("## ⏰ Temporal Robustness Analysis\n")
            temporal_results = self.results['temporal_robustness']
            
            # Hourly performance summary
            report.append("### Hourly Performance Variance\n")
            hourly_values = list(temporal_results['hourly_metrics'].values())
            primary_metric = temporal_results['primary_metric']
            hourly_scores = [h[primary_metric] for h in hourly_values]
            
            report.append(f"- **Hourly Variance:** {temporal_results['temporal_variance']['hourly_variance']:.6f}\n")
            report.append(f"- **Best Hour Performance:** {min(hourly_scores):.4f}\n") 
            report.append(f"- **Worst Hour Performance:** {max(hourly_scores):.4f}\n\n")
            
            # Generate temporal visualization
            temporal_viz_path = output_path / "temporal_performance.png"
            self._create_temporal_visualization(temporal_results, temporal_viz_path)
            generated_files['temporal_visualization'] = str(temporal_viz_path)
        
        # Stability Analysis Section
        if 'stability_analysis' in self.results:
            report.append("## 📈 Prediction Stability Analysis\n")
            stability_results = self.results['stability_analysis']
            stability_stats = stability_results['stability_statistics']
            
            report.append(f"- **Mean Performance:** {stability_stats['mean_performance']:.4f}\n")
            report.append(f"- **Performance Std Dev:** {stability_stats['std_performance']:.4f}\n")
            report.append(f"- **Coefficient of Variation:** {stability_stats['cv_performance']:.4f}\n")
            report.append(f"- **Performance Trend Slope:** {stability_stats['trend_slope']:.6f}\n\n")
            
            # Generate stability visualization  
            stability_viz_path = output_path / "stability_analysis.png"
            self._create_stability_visualization(stability_results, stability_viz_path)
            generated_files['stability_visualization'] = str(stability_viz_path)
        
        # Extreme Events Section
        if 'extreme_events' in self.results:
            report.append("## ⚡ Extreme Events Analysis\n") 
            extreme_results = self.results['extreme_events']
            
            report.append("### Performance During Extreme Events\n")
            for category in ['low_extreme', 'normal', 'high_extreme']:
                if category in extreme_results:
                    cat_data = extreme_results[category]
                    primary_metric = extreme_results['primary_metric']
                    report.append(f"- **{category.replace('_', ' ').title()}:** "
                                f"{cat_data['metrics'][primary_metric]:.4f} "
                                f"({cat_data['count']} samples, {cat_data['percentage']:.1f}%)\n")
            
            report.append("\n### Performance Degradation\n")
            for category, degradation in extreme_results['performance_degradation'].items():
                report.append(f"- **{category.replace('_', ' ').title()}:** {degradation:.2f}% degradation\n")
            report.append("\n")
        
        # Overall Robustness Summary
        report.append("## 📊 Overall Robustness Summary\n")
        robustness_score = self._calculate_overall_robustness_score()
        report.append(f"**Overall Robustness Score:** {robustness_score:.3f}/1.000\n")
        report.append(self._interpret_robustness_score(robustness_score))
        
        # Save report
        report_text = "".join(report)
        report_path = output_path / "robustness_report.md"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        generated_files['report'] = str(report_path)
        
        logger.info(f"Robustness report generated: {report_path}")
        return generated_files
    
    def _calculate_overall_robustness_score(self) -> float:
        """Calculate overall robustness score (0-1, higher is better)"""
        scores = []
        
        # Spatial consistency (lower CV is better)
        if 'spatial_robustness' in self.results:
            spatial_cv = self.results['spatial_robustness']['metric_statistics']['cv']
            spatial_score = max(0, 1 - spatial_cv)  # Invert CV
            scores.append(spatial_score)
        
        # Temporal stability (lower variance is better)
        if 'temporal_robustness' in self.results:
            hourly_var = self.results['temporal_robustness']['temporal_variance']['hourly_variance']
            temporal_score = max(0, 1 - (hourly_var * 10))  # Scale and invert
            scores.append(temporal_score)
        
        # Prediction stability (lower CV is better)
        if 'stability_analysis' in self.results:
            stability_cv = self.results['stability_analysis']['stability_statistics']['cv_performance']
            stability_score = max(0, 1 - stability_cv)
            scores.append(stability_score)
        
        return np.mean(scores) if scores else 0.0
    
    def _interpret_robustness_score(self, score: float) -> str:
        """Provide interpretation of robustness score"""
        if score >= 0.8:
            return "🟢 **Excellent robustness** - Model performs consistently across dimensions\n"
        elif score >= 0.6:
            return "🟡 **Good robustness** - Model is generally stable with some variability\n"
        elif score >= 0.4:
            return "🟠 **Moderate robustness** - Significant performance variations detected\n"
        else:
            return "🔴 **Poor robustness** - Model shows high variability across conditions\n"
    
    def _create_spatial_visualization(self, spatial_results: Dict, output_path: Path):
        """Create spatial performance visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Performance by region bar chart
        regions = list(spatial_results['region_metrics'].keys())
        primary_metric = spatial_results['primary_metric']
        performance = [spatial_results['region_metrics'][r][primary_metric] for r in regions]
        
        ax1.bar(regions, performance)
        ax1.set_title(f'{primary_metric.upper()} by Region')
        ax1.set_ylabel(primary_metric.upper())
        ax1.tick_params(axis='x', rotation=45)
        
        # Sample count by region
        counts = [spatial_results['region_counts'][r] for r in regions]
        ax2.bar(regions, counts, color='orange')
        ax2.set_title('Sample Count by Region')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_temporal_visualization(self, temporal_results: Dict, output_path: Path):
        """Create temporal performance visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        primary_metric = temporal_results['primary_metric']
        
        # Hourly performance
        hours = list(temporal_results['hourly_metrics'].keys())
        hourly_perf = [temporal_results['hourly_metrics'][h][primary_metric] for h in hours]
        ax1.plot(hours, hourly_perf, marker='o')
        ax1.set_title('Performance by Hour of Day')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel(primary_metric.upper())
        ax1.grid(True)
        
        # Day of week performance
        dow_names = list(temporal_results['day_of_week_metrics'].keys())
        dow_perf = [temporal_results['day_of_week_metrics'][d][primary_metric] for d in dow_names]
        ax2.bar(dow_names, dow_perf)
        ax2.set_title('Performance by Day of Week')
        ax2.set_ylabel(primary_metric.upper())
        ax2.tick_params(axis='x', rotation=45)
        
        # Weekend vs Weekday
        ww_names = list(temporal_results['weekend_weekday_metrics'].keys())
        ww_perf = [temporal_results['weekend_weekday_metrics'][w][primary_metric] for w in ww_names]
        ax3.bar(ww_names, ww_perf, color=['skyblue', 'lightcoral'])
        ax3.set_title('Weekday vs Weekend Performance')
        ax3.set_ylabel(primary_metric.upper())
        
        # Monthly performance
        months = list(temporal_results['monthly_metrics'].keys())
        monthly_perf = [temporal_results['monthly_metrics'][m][primary_metric] for m in months]
        ax4.bar(months, monthly_perf, color='green', alpha=0.7)
        ax4.set_title('Performance by Month')
        ax4.set_ylabel(primary_metric.upper())
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_stability_visualization(self, stability_results: Dict, output_path: Path):
        """Create stability analysis visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        rolling_df = stability_results['rolling_metrics']
        primary_metric = stability_results['primary_metric']
        
        # Performance over time
        ax1.plot(rolling_df.index, rolling_df[primary_metric], marker='o', alpha=0.7)
        ax1.set_title(f'Rolling {primary_metric.upper()} Performance Over Time')
        ax1.set_ylabel(primary_metric.upper())
        ax1.grid(True)
        
        # Performance distribution
        ax2.hist(rolling_df[primary_metric], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(rolling_df[primary_metric].mean(), color='red', linestyle='--', 
                   label=f'Mean: {rolling_df[primary_metric].mean():.4f}')
        ax2.set_title(f'{primary_metric.upper()} Distribution')
        ax2.set_xlabel(primary_metric.upper())
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    logger.info("Robustness Evaluation Framework - Ready for analysis")
    print("Available analyses: Spatial, Temporal, Stability, Extreme Events")
    print("Use RobustnessEvaluator class to perform comprehensive robustness evaluation")