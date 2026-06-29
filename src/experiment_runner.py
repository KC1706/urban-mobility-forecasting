#!/usr/bin/env python3
"""
Experiment Runner for Urban Mobility Forecasting

Main execution script that orchestrates the complete pipeline:
1. Baseline model training and evaluation (RF, XGBoost, LSTM)
2. Robustness analysis across spatial and temporal dimensions
3. LLM interpretability layer for explanations and insights

This addresses the professor's requirements for:
- Reproducing baseline results
- Robustness evaluation 
- LLM integration for interpretability
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our modules
from data_processor import MobilityDataProcessor
from baseline_models import BaselineModelTrainer, compare_baseline_models
from robustness_eval import RobustnessEvaluator

# LLM modules - imported conditionally
try:
    from llm_interpreter import LLMInterpreter, InterpretationPipeline
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Main experiment runner for urban mobility forecasting
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize experiment runner
        
        Args:
            config: Configuration dictionary for experiment parameters
        """
        self.config = config or self._default_config()
        self.results = {}
        self.experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create results directory
        self.results_dir = Path(self.config['output_dir']) / self.experiment_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Experiment initialized: {self.experiment_id}")
        logger.info(f"Results will be saved to: {self.results_dir}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for experiments"""
        return {
            'output_dir': 'results',
            'data_file': 'data/processed/chicago_taxi_processed.csv',
            'target_column': 'trip_count',
            'test_size': 0.2,
            'random_state': 42,
            'models_to_train': ['random_forest', 'xgboost', 'lstm'],
            'llm_provider': 'openai',
            'llm_model': 'gpt-4',
            'run_robustness': True,
            'run_interpretability': True,
            'spatial_column': 'pickup_borough',
            'datetime_column': 'pickup_datetime'
        }
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for experiments"""
        logger.info(f"Loading data from: {self.config['data_file']}")
        
        data_path = Path(self.config['data_file'])
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Basic validation
        required_columns = [self.config['target_column']]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert datetime column if present
        if self.config['datetime_column'] in df.columns:
            df[self.config['datetime_column']] = pd.to_datetime(df[self.config['datetime_column']], format='mixed')
        
        logger.info("Data loaded and validated successfully")
        return df
    
    def run_baseline_experiments(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run baseline model experiments
        
        Args:
            data: Prepared dataset
            
        Returns:
            Dictionary with baseline model results
        """
        logger.info("Starting baseline model experiments...")
        
        # Initialize trainer
        trainer = BaselineModelTrainer(
            task_type=self.config.get('task_type', 'regression'),
            random_state=self.config['random_state']
        )
        
        # Train models
        training_results = trainer.train_all_models(
            data=data,
            target_col=self.config['target_column'],
            models_to_train=self.config['models_to_train']
        )
        
        # Prepare test data for evaluation
        test_data = self._prepare_test_data(data, trainer)
        
        # Evaluate models
        evaluation_results = {}
        for model_name in training_results.keys():
            if 'error' not in training_results[model_name]:
                try:
                    # Get test predictions
                    predictions = self._get_model_predictions(model_name, trainer, test_data)
                    
                    # Calculate evaluation metrics
                    metrics = trainer.evaluate_model(
                        model_name, test_data['X_test'], test_data['y_test']
                    )
                    
                    evaluation_results[model_name] = {
                        'training_results': training_results[model_name],
                        'test_metrics': metrics,
                        'predictions': predictions
                    }
                    
                    logger.info(f"Model {model_name} evaluated successfully")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {str(e)}")
                    evaluation_results[model_name] = {'error': str(e)}
        
        # Create comparison summary
        comparison_df = compare_baseline_models(training_results, self.config.get('task_type', 'regression'))
        
        baseline_results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'model_comparison': comparison_df,
            'test_data': test_data
        }
        
        # Save baseline results
        self._save_baseline_results(baseline_results)
        
        logger.info("Baseline experiments completed")
        return baseline_results
    
    def _prepare_test_data(self, data: pd.DataFrame, trainer: BaselineModelTrainer) -> Dict[str, Any]:
        """Prepare test data for evaluation"""
        
        # For LSTM, we need time series splitting
        if 'lstm' in self.config['models_to_train'] and self.config['datetime_column'] in data.columns:
            # Sort by time for proper time series split
            data_sorted = data.sort_values(self.config['datetime_column'])
            
            # Prepare LSTM data separately
            lstm_data = trainer.prepare_lstm_data(
                data_sorted, 
                self.config['target_column'],
                sequence_length=24,
                test_size=self.config['test_size']
            )
            
            # Prepare regular ML data
            X, y = trainer.prepare_features(data, self.config['target_column'])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['test_size'], 
                random_state=self.config['random_state']
            )
            
            return {
                'X_train': X_train,
                'X_test': X_test, 
                'y_train': y_train,
                'y_test': y_test,
                'lstm_data': lstm_data,
                'test_indices': np.arange(len(X_test))
            }
        else:
            # Standard train-test split
            X, y = trainer.prepare_features(data, self.config['target_column'])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'test_indices': np.arange(len(X_test))
            }
    
    def _get_model_predictions(self, model_name: str, trainer: BaselineModelTrainer, 
                             test_data: Dict[str, Any]) -> np.ndarray:
        """Get predictions from trained model"""
        
        model = trainer.models[model_name]
        
        if model_name == 'lstm':
            # LSTM predictions handled differently
            _, X_test_lstm, _, _ = test_data['lstm_data']
            
            # Scale data if scaler is available
            if 'lstm' in trainer.scalers:
                scalers = trainer.scalers['lstm']
                scaler_X = scalers['scaler_X']
                scaler_y = scalers['scaler_y']
                
                # Reshape and scale
                n_samples, n_timesteps, n_features = X_test_lstm.shape
                X_test_scaled = scaler_X.transform(X_test_lstm.reshape(-1, n_features))
                X_test_scaled = X_test_scaled.reshape(X_test_lstm.shape)
                
                # Get predictions and inverse transform
                predictions_scaled = model.predict(X_test_scaled)
                predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
            else:
                predictions = model.predict(X_test_lstm).flatten()
                
        elif model_name in trainer.scalers:
            # Models with scaling (neural networks)
            scaler = trainer.scalers[model_name]
            X_test_scaled = scaler.transform(test_data['X_test'])
            predictions = model.predict(X_test_scaled)
        else:
            # Standard models
            predictions = model.predict(test_data['X_test'])
        
        return predictions
    
    def run_robustness_analysis(self, data: pd.DataFrame, 
                              baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run robustness analysis across multiple dimensions
        
        Args:
            data: Original dataset
            baseline_results: Results from baseline experiments
            
        Returns:
            Dictionary with robustness analysis results
        """
        logger.info("Starting robustness analysis...")
        
        # Initialize robustness evaluator
        evaluator = RobustnessEvaluator(task_type=self.config.get('task_type', 'regression'))
        
        # Get best model predictions for robustness analysis
        best_model_name, predictions = self._select_best_model_predictions(baseline_results)
        logger.info(f"Using {best_model_name} for robustness analysis")
        
        # Get test data indices for filtering the original data
        test_data = baseline_results['test_data']
        test_indices = test_data.get('test_indices', [])
        
        # If we have test indices, use only the test portion of data for robustness
        if len(test_indices) > 0 and len(test_indices) == len(predictions):
            robustness_data = data.iloc[test_indices].copy()
            logger.info(f"Using test set for robustness analysis: {len(robustness_data)} samples")
        else:
            # Fallback: use predictions to generate full dataset predictions
            logger.info("Generating predictions for full dataset for robustness analysis")
            robustness_data = data.copy()
            
            # Generate predictions for full dataset
            trainer = BaselineModelTrainer(
                task_type=self.config.get('task_type', 'regression'),
                random_state=self.config['random_state']
            )
            X, y = trainer.prepare_features(data, self.config['target_column'])
            
            # Get the trained model
            trained_model = baseline_results['evaluation_results'][best_model_name]['training_results']['model']
            
            # Generate predictions for full dataset
            if best_model_name in ['neural_network'] and best_model_name in trainer.scalers:
                scaler = trainer.scalers[best_model_name]
                X_scaled = scaler.transform(X)
                predictions = trained_model.predict(X_scaled)
            else:
                predictions = trained_model.predict(X)
        
        robustness_results = {}
        
        # Spatial robustness
        if (self.config['spatial_column'] in robustness_data.columns and 
            self.config.get('run_spatial_robustness', True)):
            
            logger.info("Running spatial robustness analysis...")
            spatial_results = evaluator.spatial_robustness_analysis(
                robustness_data, predictions, self.config['spatial_column']
            )
            robustness_results['spatial_robustness'] = spatial_results
        
        # Temporal robustness  
        if (self.config['datetime_column'] in robustness_data.columns and
            self.config.get('run_temporal_robustness', True)):
            
            logger.info("Running temporal robustness analysis...")
            temporal_results = evaluator.temporal_robustness_analysis(
                robustness_data, predictions, self.config['datetime_column']
            )
            robustness_results['temporal_robustness'] = temporal_results
        
        # Stability analysis
        if (self.config['datetime_column'] in robustness_data.columns and
            self.config.get('run_stability_analysis', True)):
            
            logger.info("Running stability analysis...")
            stability_results = evaluator.stability_analysis(
                robustness_data, predictions, self.config['datetime_column']
            )
            robustness_results['stability_analysis'] = stability_results
        
        # Extreme events analysis
        if self.config.get('run_extreme_events', True):
            logger.info("Running extreme events analysis...")
            extreme_results = evaluator.extreme_events_analysis(
                robustness_data, predictions, self.config['target_column']
            )
            robustness_results['extreme_events'] = extreme_results
        
        # Generate robustness report
        robustness_output_dir = self.results_dir / "robustness"
        report_files = evaluator.generate_robustness_report(str(robustness_output_dir))
        robustness_results['report_files'] = report_files
        
        # Save robustness results
        self._save_robustness_results(robustness_results)
        
        logger.info("Robustness analysis completed")
        return robustness_results
    
    def _select_best_model_predictions(self, baseline_results: Dict[str, Any]) -> tuple:
        """Select best performing model for robustness analysis"""
        
        evaluation_results = baseline_results['evaluation_results']
        
        # Find best model based on primary metric
        best_model = None
        best_score = float('inf') if self.config.get('task_type', 'regression') == 'regression' else 0
        
        for model_name, results in evaluation_results.items():
            if 'error' in results:
                continue
                
            metrics = results.get('test_metrics', {})
            
            if self.config.get('task_type', 'regression') == 'regression':
                # Lower RMSE is better
                score = metrics.get('rmse', float('inf'))
                if score < best_score:
                    best_score = score
                    best_model = model_name
            else:
                # Higher F1 is better
                score = metrics.get('f1', 0)
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        if best_model is None:
            raise ValueError("No valid model found for robustness analysis")
        
        predictions = evaluation_results[best_model]['predictions']
        return best_model, predictions
    
    def run_llm_interpretation(self, data: pd.DataFrame,
                             baseline_results: Dict[str, Any],
                             robustness_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run LLM interpretation analysis
        
        Args:
            data: Original dataset
            baseline_results: Results from baseline experiments
            robustness_results: Results from robustness analysis
            
        Returns:
            Dictionary with LLM interpretations
        """
        if not LLM_AVAILABLE:
            logger.warning("LLM modules not available - skipping interpretation")
            return {'error': 'LLM modules not available'}
            
        logger.info("Starting LLM interpretation...")
        
        # Initialize LLM interpreter
        try:
            interpreter = LLMInterpreter(
                provider=self.config['llm_provider'],
                model=self.config['llm_model']
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM interpreter: {str(e)}")
            return {'error': f"LLM initialization failed: {str(e)}"}
        
        # Initialize interpretation pipeline
        pipeline = InterpretationPipeline(interpreter)
        
        # Prepare model results for interpretation
        model_results = self._prepare_model_results_for_llm(data, baseline_results)
        
        # Run full interpretation pipeline
        try:
            interpretations = pipeline.run_full_interpretation(
                model_results=model_results,
                data=data,
                robustness_results=robustness_results
            )
            
            # Create interpretation dashboard
            interpretation_output_dir = self.results_dir / "interpretations"
            dashboard_files = interpreter.create_interpretation_dashboard(str(interpretation_output_dir))
            interpretations['dashboard_files'] = dashboard_files
            
            # Save interpretations
            self._save_interpretation_results(interpretations)
            
            logger.info("LLM interpretation completed")
            return interpretations
            
        except Exception as e:
            logger.error(f"LLM interpretation failed: {str(e)}")
            return {'error': f"LLM interpretation failed: {str(e)}"}
    
    def _prepare_model_results_for_llm(self, data: pd.DataFrame, 
                                     baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare model results for LLM interpretation"""
        
        # Get best model predictions
        best_model_name, predictions = self._select_best_model_predictions(baseline_results)
        
        # Create patterns summary
        patterns_summary = self._create_patterns_summary(data)
        
        # Create model performance summary
        model_performance = {}
        for model_name, results in baseline_results['evaluation_results'].items():
            if 'test_metrics' in results:
                model_performance[model_name] = results['test_metrics']
        
        return {
            'predictions': predictions,
            'best_model': best_model_name,
            'model_performance': model_performance,
            'patterns_summary': patterns_summary
        }
    
    def _create_patterns_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create summary of patterns in the data"""
        
        summary = {}
        
        if self.config['datetime_column'] in data.columns:
            # Extract temporal features
            dt_col = data[self.config['datetime_column']]
            target_col = data[self.config['target_column']]
            
            # Hourly patterns
            hourly_avg = data.groupby(dt_col.dt.hour)[self.config['target_column']].mean()
            summary['hourly_patterns'] = {
                'peak_hour': hourly_avg.idxmax(),
                'min_hour': hourly_avg.idxmin(), 
                'peak_demand': hourly_avg.max(),
                'min_demand': hourly_avg.min()
            }
            
            # Weekly patterns
            dow_avg = data.groupby(dt_col.dt.dayofweek)[self.config['target_column']].mean()
            summary['weekly_patterns'] = {
                'busiest_day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][dow_avg.idxmax()],
                'quietest_day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][dow_avg.idxmin()],
                'weekend_ratio': dow_avg[[5, 6]].mean() / dow_avg[[0, 1, 2, 3, 4]].mean()
            }
            
            # Basic trend analysis
            summary['seasonal_trends'] = {
                'data_span_days': (dt_col.max() - dt_col.min()).days
            }
        
        return summary
    
    def run_full_experiment(self, data_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete experiment pipeline
        
        Args:
            data_file: Optional override for data file path
            
        Returns:
            Dictionary with all experiment results
        """
        logger.info(f"Starting full experiment: {self.experiment_id}")
        
        if data_file:
            self.config['data_file'] = data_file
        
        try:
            # 1. Load data
            data = self.load_data()
            
            # 2. Run baseline experiments
            baseline_results = self.run_baseline_experiments(data)
            self.results['baseline'] = baseline_results
            
            # 3. Run robustness analysis (if enabled)
            if self.config['run_robustness']:
                robustness_results = self.run_robustness_analysis(data, baseline_results)
                self.results['robustness'] = robustness_results
            else:
                robustness_results = None
            
            # 4. Run LLM interpretation (if enabled)
            if self.config['run_interpretability']:
                interpretation_results = self.run_llm_interpretation(
                    data, baseline_results, robustness_results
                )
                self.results['interpretability'] = interpretation_results
            
            # 5. Generate final experiment summary
            experiment_summary = self._generate_experiment_summary()
            self.results['experiment_summary'] = experiment_summary
            
            # Save complete results
            self._save_experiment_results()
            
            logger.info(f"Full experiment completed: {self.experiment_id}")
            return self.results
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            self.results['error'] = str(e)
            return self.results
    
    def _generate_experiment_summary(self) -> Dict[str, Any]:
        """Generate summary of experiment results"""
        
        summary = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'components_completed': []
        }
        
        if 'baseline' in self.results:
            summary['components_completed'].append('baseline_models')
            
            # Add model performance summary
            baseline_results = self.results['baseline']
            if 'model_comparison' in baseline_results:
                summary['best_models'] = baseline_results['model_comparison'].to_dict('records')
        
        if 'robustness' in self.results:
            summary['components_completed'].append('robustness_analysis')
        
        if 'interpretability' in self.results:
            summary['components_completed'].append('llm_interpretation')
        
        summary['files_generated'] = list(self.results_dir.rglob('*'))
        
        return summary
    
    def _save_baseline_results(self, results: Dict[str, Any]):
        """Save baseline model results"""
        baseline_dir = self.results_dir / "baseline"
        baseline_dir.mkdir(exist_ok=True)
        
        # Save model comparison
        if 'model_comparison' in results:
            comparison_path = baseline_dir / "model_comparison.csv"
            results['model_comparison'].to_csv(comparison_path, index=False)
        
        # Save detailed results
        results_path = baseline_dir / "baseline_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2, default=str)
    
    def _save_robustness_results(self, results: Dict[str, Any]):
        """Save robustness analysis results"""
        robustness_path = self.results_dir / "robustness_results.json"
        with open(robustness_path, 'w') as f:
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2, default=str)
    
    def _save_interpretation_results(self, results: Dict[str, Any]):
        """Save LLM interpretation results"""
        interpretation_path = self.results_dir / "interpretation_results.json"
        with open(interpretation_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _save_experiment_results(self):
        """Save complete experiment results"""
        experiment_path = self.results_dir / "experiment_results.json"
        with open(experiment_path, 'w') as f:
            serializable_results = self._make_json_serializable(self.results)
            json.dump(serializable_results, f, indent=2, default=str)
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        else:
            return obj


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Urban Mobility Forecasting Experiments')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--models', nargs='+', 
                       choices=['random_forest', 'xgboost', 'lightgbm', 'neural_network', 'lstm'],
                       default=['random_forest', 'xgboost', 'lstm'],
                       help='Models to train')
    parser.add_argument('--no-robustness', action='store_true', help='Skip robustness analysis')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM interpretation')
    parser.add_argument('--llm-provider', choices=['openai', 'anthropic'], default='openai',
                       help='LLM provider to use')
    parser.add_argument('--llm-model', type=str, help='LLM model to use')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'output_dir': args.output,
        'models_to_train': args.models,
        'run_robustness': not args.no_robustness,
        'run_interpretability': not args.no_llm,
        'llm_provider': args.llm_provider
    }
    
    if args.llm_model:
        config['llm_model'] = args.llm_model
    
    if args.config:
        # Load config from file
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    # Initialize and run experiment
    runner = ExperimentRunner(config)
    results = runner.run_full_experiment(args.data)
    
    if 'error' in results:
        logger.error(f"Experiment failed: {results['error']}")
        return 1
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Experiment ID: {runner.experiment_id}")
    print(f"Results saved to: {runner.results_dir}")
    
    # Print summary
    if 'experiment_summary' in results:
        summary = results['experiment_summary']
        print(f"\nComponents completed: {', '.join(summary['components_completed'])}")
    
    return 0


if __name__ == "__main__":
    exit(main())