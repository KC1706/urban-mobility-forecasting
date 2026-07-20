#!/usr/bin/env python3
"""
Urban Mobility Forecasting Pipeline
Main execution script focusing on professor's requirements:
1. Baseline model reproduction (RF, XGBoost, LSTM)
2. Robustness evaluation across spatial/temporal dimensions
3. LLM interpretability layer for explanations
"""

import sys
import os
import argparse
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiment_runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("URBAN MOBILITY FORECASTING PIPELINE")
    print("Baseline Models + Robustness + LLM Interpretability")
    print("="*70 + "\n")
    
    parser = argparse.ArgumentParser(description='Urban Mobility Forecasting Pipeline')
    
    # Execution modes
    parser.add_argument('--mode', choices=['baseline', 'robustness', 'interpretability', 'full'], 
                       default='full', help='Execution mode')
    
    # Data configuration
    parser.add_argument('--data', type=str, default='data/processed/chicago_taxi_processed.csv',
                       help='Path to processed data file')
    parser.add_argument('--config', type=str, help='Path to experiment configuration file')
    
    # Model selection
    parser.add_argument('--models', nargs='+', 
                       choices=['random_forest', 'xgboost', 'lightgbm', 'neural_network', 'lstm'],
                       default=['random_forest', 'xgboost', 'lstm'],
                       help='Models to train')
    
    # LLM configuration
    parser.add_argument('--llm-provider', choices=['openai', 'anthropic'], default='openai',
                       help='LLM provider for interpretability')
    parser.add_argument('--llm-model', type=str, help='LLM model name')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM interpretation')
    
    # Output configuration
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Configure experiment based on mode
    config = {
        'output_dir': args.output,
        'models_to_train': args.models,
        'llm_provider': args.llm_provider,
        'run_robustness': args.mode in ['robustness', 'full'],
        'run_interpretability': args.mode in ['interpretability', 'full'] and not args.no_llm,
        # Essential parameters
        'target_column': 'trip_count',
        'spatial_column': 'pickup_borough',
        'datetime_column': 'pickup_datetime',
        'test_size': 0.2,
        'random_state': 42
    }
    
    if args.llm_model:
        config['llm_model'] = args.llm_model
        
    if args.config:
        import json
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    # Print configuration
    print(f"🎯 Mode: {args.mode}")
    print(f"📊 Models: {', '.join(args.models)}")
    if not args.no_llm:
        print(f"🤖 LLM: {args.llm_provider}")
    else:
        print("🤖 LLM: Disabled")
    print(f"📁 Output: {args.output}")
    print("\n" + "-"*50)
    
    try:
        # Initialize and run experiment
        runner = ExperimentRunner(config)
        results = runner.run_full_experiment(args.data)
        
        if 'error' in results:
            logger.error(f"Pipeline failed: {results['error']}")
            return 1
        
        # Print success summary
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📋 Experiment ID: {runner.experiment_id}")
        print(f"📁 Results: {runner.results_dir}")
        
        # Components completed
        if 'experiment_summary' in results:
            summary = results['experiment_summary']
            print(f"✅ Components: {', '.join(summary['components_completed'])}")
        
        # Key results — report held-out TEST metrics (not CV score)
        if 'baseline' in results and 'evaluation_results' in results['baseline']:
            print("\n🏆 Model Performance (held-out test set):")
            for model_name, res in results['baseline']['evaluation_results'].items():
                if 'error' in res:
                    print(f"  • {model_name}: ERROR ({res['error']})")
                    continue
                m = res.get('test_metrics', {})
                print(f"  • {model_name}: RMSE={m.get('rmse', float('nan')):.3f} "
                      f"MAE={m.get('mae', float('nan')):.3f} "
                      f"R²={m.get('r2', float('nan')):.4f} "
                      f"MAPE={m.get('mape', float('nan')):.1f}%")
        
        print("\n📖 Next Steps:")
        print("1. Review model performance in baseline results")
        print("2. Check robustness analysis for spatial/temporal patterns")
        if not args.no_llm:
            print("3. Read LLM interpretations for actionable insights")
        print("4. Present findings to professor focusing on robustness improvements")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        print(f"\n❌ Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
