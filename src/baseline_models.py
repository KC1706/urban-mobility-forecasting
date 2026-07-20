#!/usr/bin/env python3
"""
Baseline Models for Urban Mobility Forecasting

Implements standard models used in taxi demand prediction research:
- Random Forest
- XGBoost  
- LightGBM
- Neural Networks (MLP)
- LSTM (for time series)

Focus on reproducing results from existing papers and enabling fair comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
try:
    import xgboost as xgb
except Exception:
    xgb = None
    print("XGBoost not installed or libomp missing. XGBoost models unavailable.")

try:
    import lightgbm as lgb
except Exception:
    lgb = None
    print("LightGBM not installed. Models unavailable.")

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModelTrainer:
    """Unified trainer for all baseline models with consistent interface"""
    
    def __init__(self, task_type: str = "regression", random_state: int = 42):
        """
        Initialize baseline model trainer
        
        Args:
            task_type: "regression" for demand prediction, "classification" for demand categories
            random_state: For reproducible results
        """
        self.task_type = task_type
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def prepare_features(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for model training
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Exclude datetime columns (they should be converted to numeric features separately)
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            logger.info(f"Excluding datetime columns: {list(datetime_cols)}")
            X = X.drop(columns=datetime_cols)
        
        # Handle categorical variables (non-datetime objects)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X.values, y.values
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train Random Forest model with hyperparameter tuning"""
        
        logger.info("Training Random Forest...")
        
        # Default hyperparameters (simplified for efficiency)
        default_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [2, 5],
            'random_state': [self.random_state]
        }
        
        param_grid = kwargs.get('param_grid', default_params)
        
        # Choose model based on task type
        if self.task_type == "regression":
            model = RandomForestRegressor(random_state=self.random_state)
        else:
            model = RandomForestClassifier(random_state=self.random_state)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error' if self.task_type == "regression" else 'f1_macro',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.models['random_forest'] = grid_search.best_estimator_
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'feature_importance': grid_search.best_estimator_.feature_importances_
        }
    
    def train_xgboost(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train XGBoost model with hyperparameter tuning"""
        
        logger.info("Training XGBoost...")
        
        # Default hyperparameters (simplified)
        default_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.2],
            'random_state': [self.random_state]
        }
        
        param_grid = kwargs.get('param_grid', default_params)
        
        # Choose model based on task type  
        if self.task_type == "regression":
            model = xgb.XGBRegressor(random_state=self.random_state)
        else:
            model = xgb.XGBClassifier(random_state=self.random_state)
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error' if self.task_type == "regression" else 'f1_macro',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.models['xgboost'] = grid_search.best_estimator_
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'feature_importance': grid_search.best_estimator_.feature_importances_
        }
    
    def train_lightgbm(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train LightGBM model"""
        
        logger.info("Training LightGBM...")
        
        default_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 0.9, 1.0],
            'random_state': [self.random_state]
        }
        
        param_grid = kwargs.get('param_grid', default_params)
        
        if self.task_type == "regression":
            model = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
        else:
            model = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error' if self.task_type == "regression" else 'f1_macro',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.models['lightgbm'] = grid_search.best_estimator_
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'feature_importance': grid_search.best_estimator_.feature_importances_
        }
    
    def train_neural_network(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train Multi-Layer Perceptron"""
        
        logger.info("Training Neural Network (MLP)...")
        
        # Scale features for neural networks
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['neural_network'] = scaler
        
        default_params = {
            'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
            'activation': ['relu', 'tanh'],
            'learning_rate': ['adaptive'],
            'max_iter': [500],
            'random_state': [self.random_state]
        }
        
        param_grid = kwargs.get('param_grid', default_params)
        
        if self.task_type == "regression":
            model = MLPRegressor(random_state=self.random_state)
        else:
            model = MLPClassifier(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error' if self.task_type == "regression" else 'f1_macro',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        self.models['neural_network'] = grid_search.best_estimator_
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'scaler': scaler
        }
    
    def prepare_lstm_data(self, data: pd.DataFrame, target_col: str,
                         sequence_length: int = 24, test_size: float = 0.2,
                         group_col: str = 'pickup_borough') -> Tuple:
        """
        Prepare data for LSTM training (time series format).

        Builds look-back sequences *per spatial group* (borough/zone) so that a
        single window never spans two zones, then applies a temporal train/test
        split. Categorical and datetime columns are numerically encoded the same
        way as prepare_features() — this fixes the previous failure where the
        string column `pickup_borough` reached StandardScaler as object dtype.

        Args:
            data: Time series data (one row per timestamp x zone).
            target_col: Target column name.
            sequence_length: Number of time steps to look back.
            test_size: Proportion of *each group's* sequences held out (temporal tail).
            group_col: Column identifying the spatial unit; sequences never cross it.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        df = data.copy()

        # Sort chronologically within each spatial group.
        sort_cols = [c for c in [group_col, 'pickup_datetime'] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        # Feature columns: everything except the target and the raw datetime.
        feature_cols = [c for c in df.columns if c not in (target_col, 'pickup_datetime')]

        # Drop any remaining datetime columns (kept as engineered numeric features only).
        datetime_cols = df[feature_cols].select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            logger.info(f"LSTM: excluding datetime columns {list(datetime_cols)}")
            feature_cols = [c for c in feature_cols if c not in datetime_cols]

        # Label-encode categorical/object columns (mirrors prepare_features).
        categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # Handle missing values on the numeric feature matrix.
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

        # Build sequences independently per spatial group so windows don't cross zones.
        groups = [g for _, g in df.groupby(group_col, sort=False)] if group_col in df.columns else [df]

        X_train, X_test, y_train, y_test = [], [], [], []
        for g in groups:
            features = g[feature_cols].values.astype(np.float32)
            target = g[target_col].values.astype(np.float32)
            if len(features) <= sequence_length:
                continue  # not enough history in this group to form a window

            Xg, yg = [], []
            for i in range(sequence_length, len(features)):
                Xg.append(features[i - sequence_length:i])
                yg.append(target[i])
            Xg, yg = np.array(Xg), np.array(yg)

            # Temporal split within the group (earliest -> train, latest -> test).
            split_idx = int(len(Xg) * (1 - test_size))
            X_train.append(Xg[:split_idx]); X_test.append(Xg[split_idx:])
            y_train.append(yg[:split_idx]); y_test.append(yg[split_idx:])

        if not X_train:
            raise ValueError(
                f"No LSTM sequences could be built: every group has <= {sequence_length} rows."
            )

        X_train = np.concatenate(X_train); X_test = np.concatenate(X_test)
        y_train = np.concatenate(y_train); y_test = np.concatenate(y_test)

        return X_train, X_test, y_train, y_test
    
    def train_lstm(self, data: pd.DataFrame, target_col: str, **kwargs) -> Dict[str, Any]:
        """Train LSTM model for time series prediction"""
        
        logger.info("Training LSTM...")
        
        # Prepare LSTM data
        sequence_length = kwargs.get('sequence_length', 24)
        X_train, X_test, y_train, y_test = self.prepare_lstm_data(
            data, target_col, sequence_length
        )
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # Reshape for scaling
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, n_features))
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        
        X_test_scaled = scaler_X.transform(X_test.reshape(-1, n_features))
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(n_timesteps, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            BatchNormalization(),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_scaled,
            epochs=kwargs.get('epochs', 100),
            batch_size=kwargs.get('batch_size', 32),
            validation_split=0.2,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )
        
        # Evaluate on test set
        test_loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        test_predictions_scaled = model.predict(X_test_scaled)
        test_predictions = scaler_y.inverse_transform(test_predictions_scaled).flatten()
        
        self.models['lstm'] = model
        self.scalers['lstm'] = {'scaler_X': scaler_X, 'scaler_y': scaler_y}
        
        return {
            'model': model,
            'test_loss': test_loss,
            'test_predictions': test_predictions,
            'test_actual': y_test,
            'history': history.history,
            'scalers': {'scaler_X': scaler_X, 'scaler_y': scaler_y}
        }
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate trained model on test set"""
        
        model = self.models[model_name]
        
        # Make predictions
        if model_name in self.scalers and model_name != 'lstm':
            X_test_scaled = self.scalers[model_name].transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics based on task type
        if self.task_type == "regression":
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
        else:
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted')
            }
        
        self.results[model_name] = metrics
        return metrics
    
    def train_all_models(self, data: pd.DataFrame, target_col: str, 
                        models_to_train: List[str] = None) -> Dict[str, Any]:
        """
        Train all baseline models and return results
        
        Args:
            data: Training data
            target_col: Target column name
            models_to_train: List of models to train. If None, trains all.
        
        Returns:
            Dictionary with training results for each model
        """
        if models_to_train is None:
            models_to_train = ['random_forest', 'xgboost', 'lightgbm', 'neural_network', 'lstm']
        
        results = {}
        
        # Prepare standard ML data (non-LSTM models)
        if any(model in models_to_train for model in ['random_forest', 'xgboost', 'lightgbm', 'neural_network']):
            X, y = self.prepare_features(data, target_col)
        
        # Train each requested model
        for model_name in models_to_train:
            try:
                if model_name == 'random_forest':
                    results[model_name] = self.train_random_forest(X, y)
                elif model_name == 'xgboost':
                    results[model_name] = self.train_xgboost(X, y)
                elif model_name == 'lightgbm':
                    results[model_name] = self.train_lightgbm(X, y)
                elif model_name == 'neural_network':
                    results[model_name] = self.train_neural_network(X, y)
                elif model_name == 'lstm':
                    results[model_name] = self.train_lstm(data, target_col)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    
                logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results


def compare_baseline_models(results: Dict[str, Dict[str, Any]], 
                          task_type: str = "regression") -> pd.DataFrame:
    """
    Compare results from multiple baseline models
    
    Args:
        results: Results dictionary from train_all_models
        task_type: "regression" or "classification"
    
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for model_name, model_results in results.items():
        if 'error' in model_results:
            continue
            
        row = {'model': model_name}
        
        if task_type == "regression":
            if 'test_loss' in model_results:  # LSTM
                row['rmse'] = np.sqrt(model_results['test_loss'][0])
                row['mae'] = model_results['test_loss'][1]
            elif 'best_score' in model_results:  # Grid search models
                row['cv_score'] = -model_results['best_score']  # Convert back from negative
        else:
            if 'best_score' in model_results:
                row['cv_f1'] = model_results['best_score']
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


if __name__ == "__main__":
    # Example usage
    logger.info("Baseline Models Module - Ready for training")
    print("Available models: Random Forest, XGBoost, LightGBM, Neural Network, LSTM")
    print("Use BaselineModelTrainer class to train and evaluate models")