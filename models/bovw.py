"""Bag of Visual Words classifier."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from omegaconf import OmegaConf
import pickle
from typing import Dict, Any, Optional


class BoVWClassifier:
    """Bag of Visual Words classifier wrapper."""
        
    def __init__(
        self,
        classifier_type: str = 'random_forest',
        classifier_params: Optional[dict] = None,
        random_state: int = 42
    ):
        """
        Initialize BoVW classifier.
        
        Args:
            classifier_type: Type of classifier ('random_forest', 'svm', 'logistic')
            classifier_params: Parameters for classifier
            random_state: Random seed
        """
        self.classifier_type = classifier_type
        self.random_state = random_state

        # ✅ Handle None safely
        if classifier_params is None:
            classifier_params = {}

        # ✅ If OmegaConf object, convert it
        try:
            from omegaconf import OmegaConf
            if not isinstance(classifier_params, dict) and OmegaConf.is_config(classifier_params):
                classifier_params = OmegaConf.to_container(classifier_params, resolve=True)
        except ImportError:
            pass  # just in case OmegaConf isn't used

        # ✅ Initialize classifier
        params = classifier_params.to_dict() if hasattr(classifier_params, "to_dict") else dict(classifier_params)
        if classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                **params
            )
        elif classifier_type == 'svm':
            self.classifier = SVC(
                random_state=random_state,
                **params
            )
        elif classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                random_state=random_state,
                **params
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        self.is_fitted = False



    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train classifier.
        
        Args:
            X: Training histograms (n_samples, n_clusters)
            y: Training labels (n_samples,)
        """
        print(f"Training {self.classifier_type} classifier...")
        print(f"Training samples: {X.shape[0]}")
        print(f"Feature dimension: {X.shape[1]}")
        
        self.classifier.fit(X, y)
        self.is_fitted = True
        
        # Training accuracy
        train_pred = self.classifier.predict(X)
        train_acc = accuracy_score(y, train_pred)
        print(f"Training accuracy: {train_acc:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.
        
        Args:
            X: Input histograms (n_samples, n_clusters)
            
        Returns:
            Predicted labels (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        
        return self.classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input histograms (n_samples, n_clusters)
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.classifier_type} does not support predict_proba")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, class_names: Optional[list] = None) -> Dict[str, float]:
        """
        Evaluate classifier.
        
        Args:
            X: Test histograms (n_samples, n_clusters)
            y: Test labels (n_samples,)
            class_names: List of class names for report
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, predictions, target_names=class_names, digits=4))
        
        return {
            'accuracy': accuracy,
            'predictions': predictions
        }
    
    def save(self, filepath: str):
        """Save classifier to file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted classifier")
        
        save_data = {
            'classifier': self.classifier,
            'classifier_type': self.classifier_type,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✓ Classifier saved to {filepath}")
    
    def load(self, filepath: str):
        """Load classifier from file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.classifier = save_data['classifier']
        self.classifier_type = save_data['classifier_type']
        self.random_state = save_data['random_state']
        self.is_fitted = True
        
        print(f"✓ Classifier loaded from {filepath}")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance (if supported)."""
        if hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_
        return None