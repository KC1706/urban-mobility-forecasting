#!/usr/bin/env python3
"""
LLM Interpretability Layer for Urban Mobility Forecasting

This module provides LLM-based interpretations of mobility predictions, offering:
- Contextual explanations for prediction results
- Policy recommendations based on patterns
- Natural language insights about demand drivers
- Multi-modal reasoning combining numerical predictions with domain knowledge

This is the UNIQUE CONTRIBUTION that the professor liked - bridging predictions and actionable insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
import re
from datetime import datetime, timedelta

# LLM Providers
import openai
from anthropic import Anthropic
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMInterpreter:
    """
    LLM-based interpretability layer for urban mobility predictions
    """
    
    def __init__(self, provider: str = "openai", model: str = None, api_key: str = None):
        """
        Initialize LLM interpreter
        
        Args:
            provider: "openai" or "anthropic"
            model: Model name (e.g., "gpt-4", "claude-3-sonnet")
            api_key: API key (will use env variables if not provided)
        """
        self.provider = provider.lower()
        self.model = model
        
        # Set up LLM client
        if self.provider == "openai":
            self.model = model or "gpt-4"
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
            
        elif self.provider == "anthropic":
            self.model = model or "claude-3-sonnet-20240229"
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # System prompts for different interpretation tasks
        self.system_prompts = self._initialize_system_prompts()
        
        logger.info(f"LLM Interpreter initialized: {self.provider} / {self.model}")
    
    def _initialize_system_prompts(self) -> Dict[str, str]:
        """Initialize system prompts for different interpretation tasks"""
        
        return {
            'prediction_explanation': """
            You are an expert urban mobility analyst. Your task is to provide clear, insightful explanations 
            for taxi demand predictions in NYC. Given prediction results and contextual data, explain:
            
            1. WHY the model predicted this demand level
            2. WHAT factors likely contributed to this prediction
            3. HOW confident we should be in this prediction
            4. WHAT this means for transportation planning
            
            Be specific, data-driven, and actionable. Use your knowledge of NYC geography, 
            transportation patterns, and urban mobility dynamics.
            """,
            
            'pattern_analysis': """
            You are an urban transportation researcher analyzing mobility patterns. Given time series 
            data and model predictions, identify and explain:
            
            1. Key patterns in demand (temporal, spatial, seasonal)
            2. Anomalies or unexpected behaviors
            3. Potential causes and contributing factors
            4. Implications for city planning and policy
            
            Provide insights that go beyond what basic statistics show. Think about causality, 
            urban dynamics, and real-world factors affecting mobility.
            """,
            
            'robustness_interpretation': """
            You are a model evaluation expert focusing on AI reliability in urban transportation. 
            Given robustness analysis results, explain:
            
            1. Where and when the model performs well vs poorly
            2. What causes performance variations
            3. Reliability implications for different use cases
            4. Recommendations for improving model robustness
            
            Focus on practical implications for deploying this model in real-world transportation systems.
            """,
            
            'policy_recommendations': """
            You are a transportation policy advisor for NYC. Based on mobility predictions and analysis, 
            provide actionable policy recommendations:
            
            1. Infrastructure improvements
            2. Service optimization strategies  
            3. Demand management approaches
            4. Emergency preparedness considerations
            
            Ground recommendations in data while considering political feasibility, budget constraints, 
            and multi-stakeholder impacts.
            """
        }
    
    def _call_llm(self, prompt: str, system_prompt: str, max_tokens: int = 1000) -> str:
        """Make API call to LLM provider"""
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3  # Lower temperature for more consistent, factual responses
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            return f"Error generating interpretation: {str(e)}"
    
    def explain_prediction(self, prediction_data: Dict[str, Any], 
                          context_data: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate explanation for a specific prediction
        
        Args:
            prediction_data: Dictionary containing prediction info
            context_data: Additional contextual information
            
        Returns:
            Dictionary with explanation components
        """
        logger.info("Generating prediction explanation...")
        
        # Construct prompt with prediction details
        prompt = self._build_prediction_prompt(prediction_data, context_data)
        
        # Get LLM explanation
        explanation = self._call_llm(
            prompt, 
            self.system_prompts['prediction_explanation'],
            max_tokens=800
        )
        
        return {
            'explanation': explanation,
            'prediction_info': prediction_data,
            'context': context_data,
            'timestamp': datetime.now().isoformat()
        }
    
    def _build_prediction_prompt(self, prediction_data: Dict[str, Any], 
                                context_data: Dict[str, Any] = None) -> str:
        """Build prompt for prediction explanation"""
        
        prompt_parts = []
        
        # Core prediction information
        prompt_parts.append("PREDICTION DETAILS:")
        
        if 'predicted_demand' in prediction_data:
            prompt_parts.append(f"- Predicted demand: {prediction_data['predicted_demand']:.2f}")
        
        if 'actual_demand' in prediction_data:
            prompt_parts.append(f"- Actual demand: {prediction_data['actual_demand']:.2f}")
            error = abs(prediction_data['predicted_demand'] - prediction_data['actual_demand'])
            prompt_parts.append(f"- Prediction error: {error:.2f}")
        
        if 'confidence' in prediction_data:
            prompt_parts.append(f"- Model confidence: {prediction_data['confidence']:.3f}")
        
        # Location information
        if 'location' in prediction_data:
            loc = prediction_data['location']
            prompt_parts.append(f"- Location: {loc}")
        
        # Temporal information
        if 'datetime' in prediction_data:
            dt = pd.to_datetime(prediction_data['datetime'])
            prompt_parts.append(f"- Date/Time: {dt.strftime('%Y-%m-%d %H:%M')} ({dt.strftime('%A')})")
            prompt_parts.append(f"- Hour of day: {dt.hour}")
            prompt_parts.append(f"- Day of week: {dt.strftime('%A')}")
            prompt_parts.append(f"- Month: {dt.strftime('%B')}")
        
        # Add context data if available
        if context_data:
            prompt_parts.append("\nCONTEXT DATA:")
            
            if 'weather' in context_data:
                weather = context_data['weather']
                prompt_parts.append(f"- Weather: {weather}")
            
            if 'events' in context_data:
                events = context_data['events']
                prompt_parts.append(f"- Special events: {events}")
            
            if 'historical_avg' in context_data:
                avg = context_data['historical_avg']
                prompt_parts.append(f"- Historical average for this time: {avg:.2f}")
            
            if 'feature_importance' in context_data:
                importance = context_data['feature_importance']
                prompt_parts.append("- Top contributing features:")
                for feature, score in importance.items():
                    prompt_parts.append(f"  * {feature}: {score:.3f}")
        
        prompt_parts.append("\nPlease provide a comprehensive explanation for this prediction.")
        
        return "\n".join(prompt_parts)
    
    def analyze_patterns(self, time_series_data: pd.DataFrame, 
                        patterns_summary: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze and interpret patterns in mobility data
        
        Args:
            time_series_data: Time series of predictions and actuals
            patterns_summary: Summary statistics of identified patterns
            
        Returns:
            Dictionary with pattern analysis
        """
        logger.info("Analyzing mobility patterns...")
        
        prompt = self._build_patterns_prompt(time_series_data, patterns_summary)
        
        analysis = self._call_llm(
            prompt,
            self.system_prompts['pattern_analysis'],
            max_tokens=1200
        )
        
        return {
            'pattern_analysis': analysis,
            'data_summary': patterns_summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def _build_patterns_prompt(self, data: pd.DataFrame, summary: Dict[str, Any]) -> str:
        """Build prompt for pattern analysis"""
        
        prompt_parts = []
        
        # Data overview
        prompt_parts.append("MOBILITY DATA ANALYSIS:")
        prompt_parts.append(f"- Time period: {data.index.min()} to {data.index.max()}")
        prompt_parts.append(f"- Total observations: {len(data)}")
        
        if 'predicted_demand' in data.columns and 'actual_demand' in data.columns:
            avg_actual = data['actual_demand'].mean()
            avg_predicted = data['predicted_demand'].mean()
            prompt_parts.append(f"- Average actual demand: {avg_actual:.2f}")
            prompt_parts.append(f"- Average predicted demand: {avg_predicted:.2f}")
        
        # Pattern summary
        prompt_parts.append("\nIDENTIFIED PATTERNS:")
        
        if 'hourly_patterns' in summary:
            hourly = summary['hourly_patterns']
            prompt_parts.append("HOURLY PATTERNS:")
            prompt_parts.append(f"- Peak hour: {hourly.get('peak_hour', 'Unknown')}")
            prompt_parts.append(f"- Lowest demand hour: {hourly.get('min_hour', 'Unknown')}")
            prompt_parts.append(f"- Peak demand: {hourly.get('peak_demand', 0):.2f}")
        
        if 'weekly_patterns' in summary:
            weekly = summary['weekly_patterns']
            prompt_parts.append("WEEKLY PATTERNS:")
            prompt_parts.append(f"- Busiest day: {weekly.get('busiest_day', 'Unknown')}")
            prompt_parts.append(f"- Quietest day: {weekly.get('quietest_day', 'Unknown')}")
            prompt_parts.append(f"- Weekend vs weekday ratio: {weekly.get('weekend_ratio', 1.0):.2f}")
        
        if 'seasonal_trends' in summary:
            seasonal = summary['seasonal_trends']
            prompt_parts.append("SEASONAL TRENDS:")
            if 'trend_slope' in seasonal:
                slope = seasonal['trend_slope']
                trend_dir = "increasing" if slope > 0 else "decreasing"
                prompt_parts.append(f"- Overall trend: {trend_dir} ({slope:.4f})")
        
        if 'anomalies' in summary:
            anomalies = summary['anomalies']
            prompt_parts.append(f"- Detected anomalies: {len(anomalies)} time periods")
        
        prompt_parts.append("\nPlease analyze these patterns and provide insights about urban mobility dynamics.")
        
        return "\n".join(prompt_parts)
    
    def interpret_robustness(self, robustness_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Interpret robustness analysis results
        
        Args:
            robustness_results: Results from RobustnessEvaluator
            
        Returns:
            Dictionary with robustness interpretation
        """
        logger.info("Interpreting robustness analysis...")
        
        prompt = self._build_robustness_prompt(robustness_results)
        
        interpretation = self._call_llm(
            prompt,
            self.system_prompts['robustness_interpretation'],
            max_tokens=1000
        )
        
        return {
            'robustness_interpretation': interpretation,
            'analysis_summary': robustness_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _build_robustness_prompt(self, results: Dict[str, Any]) -> str:
        """Build prompt for robustness interpretation"""
        
        prompt_parts = []
        prompt_parts.append("MODEL ROBUSTNESS ANALYSIS RESULTS:")
        
        # Spatial robustness
        if 'spatial_robustness' in results:
            spatial = results['spatial_robustness']
            stats = spatial['metric_statistics']
            
            prompt_parts.append("\nSPATIAL ROBUSTNESS:")
            prompt_parts.append(f"- Performance varies by region (CV: {stats['cv']:.3f})")
            prompt_parts.append(f"- Best region performance: {stats['min']:.3f}")
            prompt_parts.append(f"- Worst region performance: {stats['max']:.3f}")
            
            # Top and bottom performers
            region_metrics = spatial['region_metrics']
            primary_metric = spatial['primary_metric']
            
            if region_metrics:
                performances = {region: metrics[primary_metric] 
                              for region, metrics in region_metrics.items()}
                best_region = min(performances.keys(), key=lambda x: performances[x])
                worst_region = max(performances.keys(), key=lambda x: performances[x])
                
                prompt_parts.append(f"- Best performing region: {best_region}")
                prompt_parts.append(f"- Worst performing region: {worst_region}")
        
        # Temporal robustness
        if 'temporal_robustness' in results:
            temporal = results['temporal_robustness']
            variance = temporal['temporal_variance']
            
            prompt_parts.append("\nTEMPORAL ROBUSTNESS:")
            prompt_parts.append(f"- Hourly performance variance: {variance['hourly_variance']:.6f}")
            prompt_parts.append(f"- Day-of-week variance: {variance['dow_variance']:.6f}")
            prompt_parts.append(f"- Monthly variance: {variance['monthly_variance']:.6f}")
        
        # Stability analysis
        if 'stability_analysis' in results:
            stability = results['stability_analysis']
            stats = stability['stability_statistics']
            
            prompt_parts.append("\nPREDICTION STABILITY:")
            prompt_parts.append(f"- Performance coefficient of variation: {stats['cv_performance']:.3f}")
            prompt_parts.append(f"- Performance trend slope: {stats['trend_slope']:.6f}")
        
        # Extreme events
        if 'extreme_events' in results:
            extreme = results['extreme_events']
            degradation = extreme.get('performance_degradation', {})
            
            prompt_parts.append("\nEXTREME EVENTS PERFORMANCE:")
            for event_type, deg in degradation.items():
                prompt_parts.append(f"- {event_type.replace('_', ' ').title()}: {deg:.1f}% degradation")
        
        prompt_parts.append("\nPlease interpret these robustness results and their implications for model deployment.")
        
        return "\n".join(prompt_parts)
    
    def generate_policy_recommendations(self, analysis_summary: Dict[str, Any], 
                                      prediction_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate policy recommendations based on analysis results
        
        Args:
            analysis_summary: Summary of all analyses performed
            prediction_results: Key prediction and performance metrics
            
        Returns:
            Dictionary with policy recommendations
        """
        logger.info("Generating policy recommendations...")
        
        prompt = self._build_policy_prompt(analysis_summary, prediction_results)
        
        recommendations = self._call_llm(
            prompt,
            self.system_prompts['policy_recommendations'],
            max_tokens=1200
        )
        
        return {
            'policy_recommendations': recommendations,
            'based_on': analysis_summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def _build_policy_prompt(self, analysis: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Build prompt for policy recommendations"""
        
        prompt_parts = []
        prompt_parts.append("URBAN MOBILITY ANALYSIS SUMMARY FOR POLICY RECOMMENDATIONS:")
        
        # Model performance overview
        if 'model_performance' in results:
            perf = results['model_performance']
            prompt_parts.append("\nMODEL PERFORMANCE:")
            for model, metrics in perf.items():
                prompt_parts.append(f"- {model}: RMSE = {metrics.get('rmse', 'N/A')}")
        
        # Key findings from analyses
        if 'key_findings' in analysis:
            findings = analysis['key_findings']
            prompt_parts.append("\nKEY FINDINGS:")
            for finding in findings:
                prompt_parts.append(f"- {finding}")
        
        # Robustness challenges
        if 'robustness_issues' in analysis:
            issues = analysis['robustness_issues']
            prompt_parts.append("\nROBUSTNESS CHALLENGES:")
            for issue in issues:
                prompt_parts.append(f"- {issue}")
        
        # Demand patterns
        if 'demand_patterns' in analysis:
            patterns = analysis['demand_patterns']
            prompt_parts.append("\nDEMAND PATTERNS:")
            for pattern in patterns:
                prompt_parts.append(f"- {pattern}")
        
        prompt_parts.append("\nBased on this analysis, provide specific, actionable policy recommendations for NYC transportation planning.")
        
        return "\n".join(prompt_parts)
    
    def create_interpretation_dashboard(self, output_dir: str = "results/interpretations") -> Dict[str, str]:
        """
        Create a comprehensive interpretation dashboard
        
        Args:
            output_dir: Directory to save interpretation results
            
        Returns:
            Dictionary with paths to generated interpretation files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dashboard_html = self._generate_dashboard_html()
        
        dashboard_path = output_path / "interpretation_dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
        
        logger.info(f"Interpretation dashboard created: {dashboard_path}")
        
        return {'dashboard': str(dashboard_path)}
    
    def _generate_dashboard_html(self) -> str:
        """Generate HTML dashboard for interpretations"""
        
        html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Urban Mobility LLM Interpretations Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .interpretation-section { 
                    border: 1px solid #ddd; 
                    margin: 20px 0; 
                    padding: 15px; 
                    border-radius: 5px;
                }
                .metric { 
                    background-color: #f5f5f5; 
                    padding: 10px; 
                    border-radius: 3px; 
                    margin: 5px 0;
                }
                .explanation { 
                    background-color: #e8f4f8; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 10px 0;
                }
                .recommendation {
                    background-color: #fff3cd;
                    padding: 15px;
                    border-left: 4px solid #ffc107;
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
            <h1>🤖 LLM Interpretability Dashboard</h1>
            <h2>Urban Mobility Forecasting - Explanations & Insights</h2>
            
            <div class="interpretation-section">
                <h3>📊 Model Explanations</h3>
                <p>This dashboard will be populated with LLM interpretations when you run the analysis pipeline.</p>
                <div class="explanation">
                    <strong>Example Interpretation:</strong><br>
                    The model predicted high demand (85 trips/hour) for Manhattan at 8 PM on Friday because:
                    <ul>
                        <li>Weekend evening rush combining with nightlife activities</li>
                        <li>Weather is clear, encouraging outdoor activities</li>
                        <li>Historical patterns show 40% higher demand on Friday evenings</li>
                        <li>Theater district events likely contributing to increased mobility</li>
                    </ul>
                </div>
            </div>
            
            <div class="interpretation-section">
                <h3>🎯 Policy Recommendations</h3>
                <div class="recommendation">
                    <strong>Dynamic Pricing Strategy:</strong> Implement surge pricing during predicted high-demand periods to better distribute demand across time and space.
                </div>
                <div class="recommendation">
                    <strong>Infrastructure Planning:</strong> Focus on improving taxi availability in consistently under-served areas identified by spatial robustness analysis.
                </div>
            </div>
            
            <div class="interpretation-section">
                <h3>🔍 Pattern Insights</h3>
                <p>LLM-generated insights about temporal and spatial patterns will appear here after analysis.</p>
            </div>
            
            <p><em>Dashboard generated by LLM Interpretability Framework</em></p>
        </body>
        </html>
        '''
        
        return html


class InterpretationPipeline:
    """
    End-to-end pipeline for generating LLM interpretations
    """
    
    def __init__(self, llm_interpreter: LLMInterpreter):
        """
        Initialize interpretation pipeline
        
        Args:
            llm_interpreter: Configured LLMInterpreter instance
        """
        self.interpreter = llm_interpreter
        self.results = {}
    
    def run_full_interpretation(self, model_results: Dict[str, Any], 
                              data: pd.DataFrame, 
                              robustness_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run complete interpretation pipeline
        
        Args:
            model_results: Results from baseline models
            data: Original data with predictions
            robustness_results: Results from robustness evaluation
            
        Returns:
            Dictionary with all interpretations
        """
        logger.info("Running full LLM interpretation pipeline...")
        
        interpretations = {}
        
        # 1. Explain key predictions
        if 'predictions' in model_results:
            logger.info("Generating prediction explanations...")
            sample_predictions = self._select_sample_predictions(data, model_results['predictions'])
            
            explanations = []
            for pred_data in sample_predictions:
                explanation = self.interpreter.explain_prediction(pred_data)
                explanations.append(explanation)
            
            interpretations['prediction_explanations'] = explanations
        
        # 2. Analyze patterns
        if 'patterns_summary' in model_results:
            logger.info("Analyzing patterns...")
            pattern_analysis = self.interpreter.analyze_patterns(
                data, model_results['patterns_summary']
            )
            interpretations['pattern_analysis'] = pattern_analysis
        
        # 3. Interpret robustness
        if robustness_results:
            logger.info("Interpreting robustness...")
            robustness_interpretation = self.interpreter.interpret_robustness(robustness_results)
            interpretations['robustness_interpretation'] = robustness_interpretation
        
        # 4. Generate policy recommendations
        logger.info("Generating policy recommendations...")
        analysis_summary = self._create_analysis_summary(model_results, robustness_results)
        policy_recs = self.interpreter.generate_policy_recommendations(
            analysis_summary, model_results
        )
        interpretations['policy_recommendations'] = policy_recs
        
        self.results = interpretations
        logger.info("LLM interpretation pipeline complete")
        
        return interpretations
    
    def _select_sample_predictions(self, data: pd.DataFrame, 
                                 predictions: np.ndarray, n_samples: int = 5) -> List[Dict]:
        """Select representative predictions for explanation"""
        
        # Select diverse samples (high, low, medium demand)
        target_col = 'trip_count' if 'trip_count' in data.columns else 'demand'
        
        if target_col not in data.columns:
            logger.warning("No target column found for sample selection")
            return []
        
        # Get percentiles for sampling
        low_threshold = np.percentile(data[target_col], 20)
        high_threshold = np.percentile(data[target_col], 80)
        
        samples = []
        
        # High demand samples
        high_mask = data[target_col] >= high_threshold
        if high_mask.sum() > 0:
            high_idx = np.random.choice(np.where(high_mask)[0], 
                                       min(2, high_mask.sum()), replace=False)
            for idx in high_idx:
                samples.append(self._create_prediction_dict(data.iloc[idx], predictions[idx]))
        
        # Low demand samples
        low_mask = data[target_col] <= low_threshold
        if low_mask.sum() > 0:
            low_idx = np.random.choice(np.where(low_mask)[0], 
                                      min(2, low_mask.sum()), replace=False)
            for idx in low_idx:
                samples.append(self._create_prediction_dict(data.iloc[idx], predictions[idx]))
        
        # Medium demand sample
        medium_mask = (data[target_col] > low_threshold) & (data[target_col] < high_threshold)
        if medium_mask.sum() > 0:
            medium_idx = np.random.choice(np.where(medium_mask)[0], 1)[0]
            samples.append(self._create_prediction_dict(data.iloc[medium_idx], predictions[medium_idx]))
        
        return samples[:n_samples]
    
    def _create_prediction_dict(self, row: pd.Series, prediction: float) -> Dict[str, Any]:
        """Create prediction dictionary for LLM interpretation"""
        
        pred_dict = {
            'predicted_demand': prediction,
        }
        
        # Add available columns
        if 'trip_count' in row:
            pred_dict['actual_demand'] = row['trip_count']
        elif 'demand' in row:
            pred_dict['actual_demand'] = row['demand']
        
        if 'pickup_datetime' in row:
            pred_dict['datetime'] = row['pickup_datetime']
        
        if 'pickup_borough' in row:
            pred_dict['location'] = row['pickup_borough']
        
        return pred_dict
    
    def _create_analysis_summary(self, model_results: Dict, robustness_results: Dict = None) -> Dict[str, Any]:
        """Create summary for policy recommendations"""
        
        summary = {
            'key_findings': [],
            'robustness_issues': [],
            'demand_patterns': []
        }
        
        # Add findings based on available results
        if model_results and 'model_performance' in model_results:
            summary['key_findings'].append("Multiple baseline models trained and evaluated")
        
        if robustness_results:
            if 'spatial_robustness' in robustness_results:
                cv = robustness_results['spatial_robustness']['metric_statistics']['cv']
                if cv > 0.3:
                    summary['robustness_issues'].append("High spatial performance variability detected")
                    
            if 'temporal_robustness' in robustness_results:
                temporal_var = robustness_results['temporal_robustness']['temporal_variance']
                if temporal_var['hourly_variance'] > 0.1:
                    summary['robustness_issues'].append("Significant hourly performance fluctuations")
        
        return summary


if __name__ == "__main__":
    logger.info("LLM Interpretability Module - Ready for interpretation")
    print("Available interpretations: Prediction explanations, Pattern analysis, Robustness interpretation, Policy recommendations")
    print("Use LLMInterpreter class to generate insights from model outputs")