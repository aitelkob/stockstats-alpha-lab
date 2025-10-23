"""
Machine learning models module for financial time series.

This module provides sklearn pipelines, XGBoost models, and evaluation
frameworks for predicting financial returns and market regimes.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPipeline:
    """Main class for creating and evaluating ML pipelines."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model pipeline.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.pipelines = {}
        self.feature_importance = {}
    
    def create_classification_pipeline(
        self,
        model_name: str,
        model_type: str = "logistic",
        feature_selection: bool = True,
        scaling: bool = True
    ) -> Pipeline:
        """
        Create a classification pipeline.
        
        Args:
            model_name: Name for the model
            model_type: Type of model ('logistic', 'random_forest', 'xgboost', 'svm')
            feature_selection: Whether to include feature selection
            scaling: Whether to include scaling
            
        Returns:
            Configured sklearn Pipeline
        """
        steps = []
        
        # Add scaling if requested
        if scaling:
            steps.append(('scaler', StandardScaler()))
        
        # Add feature selection (simplified - could use more sophisticated methods)
        if feature_selection:
            from sklearn.feature_selection import SelectKBest, f_classif
            steps.append(('feature_selection', SelectKBest(f_classif, k=20)))
        
        # Add model
        if model_type == "logistic":
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=5
            )
        elif model_type == "xgboost":
            model = xgb.XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            )
        elif model_type == "svm":
            model = SVC(random_state=self.random_state, probability=True)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        steps.append(('classifier', model))
        
        pipeline = Pipeline(steps)
        self.pipelines[model_name] = pipeline
        
        logger.info(f"Created {model_type} classification pipeline: {model_name}")
        return pipeline
    
    def create_regression_pipeline(
        self,
        model_name: str,
        model_type: str = "ridge",
        feature_selection: bool = True,
        scaling: bool = True
    ) -> Pipeline:
        """
        Create a regression pipeline.
        
        Args:
            model_name: Name for the model
            model_type: Type of model ('ridge', 'random_forest', 'xgboost', 'svr')
            feature_selection: Whether to include feature selection
            scaling: Whether to include scaling
            
        Returns:
            Configured sklearn Pipeline
        """
        steps = []
        
        # Add scaling if requested
        if scaling:
            steps.append(('scaler', StandardScaler()))
        
        # Add feature selection
        if feature_selection:
            from sklearn.feature_selection import SelectKBest, f_regression
            steps.append(('feature_selection', SelectKBest(f_regression, k=20)))
        
        # Add model
        if model_type == "ridge":
            model = Ridge(alpha=1.0, random_state=self.random_state)
        elif model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=5
            )
        elif model_type == "xgboost":
            model = xgb.XGBRegressor(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            )
        elif model_type == "svr":
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        steps.append(('regressor', model))
        
        pipeline = Pipeline(steps)
        self.pipelines[model_name] = pipeline
        
        logger.info(f"Created {model_type} regression pipeline: {model_name}")
        return pipeline
    
    def train_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train a model and return evaluation metrics.
        
        Args:
            model_name: Name of the model to train
            X: Feature matrix
            y: Target labels/values
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with training results and metrics
        """
        if model_name not in self.pipelines:
            raise ValueError(f"Model {model_name} not found")
        
        pipeline = self.pipelines[model_name]
        
        # Split data (time-series aware)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_val)
        y_pred_proba = None
        
        # Get probabilities for classification models
        if hasattr(pipeline.named_steps[list(pipeline.named_steps.keys())[-1]], 'predict_proba'):
            y_pred_proba = pipeline.predict_proba(X_val)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
        
        # Store feature importance if available
        self._extract_feature_importance(pipeline, model_name, X.columns)
        
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'feature_importance': self.feature_importance.get(model_name, {}),
            'pipeline': pipeline
        }
        
        logger.info(f"Trained {model_name}: {metrics}")
        return results
    
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Determine if classification or regression
        is_classification = y_true.dtype in ['int64', 'int32', 'bool'] or len(np.unique(y_true)) < 10
        
        if is_classification:
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
            
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            # Confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            
        else:
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
            
            # R-squared
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return metrics
    
    def _extract_feature_importance(
        self,
        pipeline: Pipeline,
        model_name: str,
        feature_names: List[str]
    ) -> None:
        """Extract feature importance from trained model."""
        try:
            # Get the final estimator
            final_estimator = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
            
            if hasattr(final_estimator, 'feature_importances_'):
                # Tree-based models
                importance = final_estimator.feature_importances_
            elif hasattr(final_estimator, 'coef_'):
                # Linear models
                importance = np.abs(final_estimator.coef_).flatten()
            else:
                return
            
            # Map to feature names (account for feature selection)
            if 'feature_selection' in pipeline.named_steps:
                selected_features = pipeline.named_steps['feature_selection'].get_support()
                feature_names = [name for name, selected in zip(feature_names, selected_features) if selected]
            
            # Ensure we have the right number of features
            if len(importance) == len(feature_names):
                self.feature_importance[model_name] = dict(zip(feature_names, importance))
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance for {model_name}: {e}")
    
    def cross_validate_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict:
        """
        Perform time-series cross-validation.
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target labels/values
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        if model_name not in self.pipelines:
            raise ValueError(f"Model {model_name} not found")
        
        pipeline = self.pipelines[model_name]
        
        # Use TimeSeriesSplit for proper time-series CV
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Determine scoring metric
        is_classification = y.dtype in ['int64', 'int32', 'bool'] or len(np.unique(y)) < 10
        scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=tscv, scoring=scoring)
        
        results = {
            'model_name': model_name,
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'scoring': scoring
        }
        
        logger.info(f"CV results for {model_name}: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        return results


def create_baseline_models() -> Dict[str, Pipeline]:
    """
    Create a set of baseline models for comparison.
    
    Returns:
        Dictionary mapping model names to pipelines
    """
    pipeline_manager = ModelPipeline()
    
    # Classification models
    pipeline_manager.create_classification_pipeline("logistic_baseline", "logistic")
    pipeline_manager.create_classification_pipeline("rf_baseline", "random_forest")
    pipeline_manager.create_classification_pipeline("xgb_baseline", "xgboost")
    
    # Regression models
    pipeline_manager.create_regression_pipeline("ridge_baseline", "ridge")
    pipeline_manager.create_regression_pipeline("rf_reg_baseline", "random_forest")
    pipeline_manager.create_regression_pipeline("xgb_reg_baseline", "xgboost")
    
    return pipeline_manager.pipelines


def compare_models(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    validation_split: float = 0.2
) -> pd.DataFrame:
    """
    Compare multiple models and return results.
    
    Args:
        models: Dictionary of model names to pipelines
        X: Feature matrix
        y: Target labels/values
        validation_split: Fraction of data to use for validation
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, pipeline in models.items():
        try:
            # Train and evaluate
            pipeline_manager = ModelPipeline()
            pipeline_manager.pipelines[model_name] = pipeline
            
            result = pipeline_manager.train_model(model_name, X, y, validation_split)
            
            # Extract key metrics
            metrics = result['metrics']
            row = {'model': model_name}
            
            if 'accuracy' in metrics:
                row.update({
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1']
                })
                if 'auc' in metrics:
                    row['auc'] = metrics['auc']
            else:
                row.update({
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2']
                })
            
            results.append(row)
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue
    
    return pd.DataFrame(results)


def calculate_information_coefficient_ml(
    y_true: pd.Series,
    y_pred: pd.Series,
    method: str = "spearman"
) -> float:
    """
    Calculate Information Coefficient for ML predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        method: Correlation method
        
    Returns:
        IC value
    """
    if method == "spearman":
        return y_true.corr(y_pred, method='spearman')
    elif method == "pearson":
        return y_true.corr(y_pred, method='pearson')
    else:
        raise ValueError(f"Unknown correlation method: {method}")


if __name__ == "__main__":
    # Example usage
    from data import DataLoader
    from indicators import add_basic_indicators
    from labeling import LabelingEngine, create_feature_matrix
    
    # Load and prepare data
    loader = DataLoader()
    df = loader.load_single_ticker("AAPL", period="1y")
    df = add_basic_indicators(df)
    
    # Create labels
    labeler = LabelingEngine()
    forward_returns = labeler.forward_return_label(df, horizon=5)
    binary_labels = labeler.binary_classification_label(forward_returns)
    
    # Create feature matrix
    indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    X, y = create_feature_matrix(df, indicator_cols, binary_labels.name)
    
    # Create and train models
    pipeline_manager = ModelPipeline()
    
    # Classification models
    pipeline_manager.create_classification_pipeline("logistic", "logistic")
    pipeline_manager.create_classification_pipeline("xgboost", "xgboost")
    
    # Train and compare
    results = []
    for model_name in ["logistic", "xgboost"]:
        result = pipeline_manager.train_model(model_name, X, y)
        results.append({
            'model': model_name,
            'accuracy': result['metrics']['accuracy'],
            'f1': result['metrics']['f1']
        })
    
    comparison_df = pd.DataFrame(results)
    print("Model comparison:")
    print(comparison_df)
