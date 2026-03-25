import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from feature_extractor import extract_features_from_image
from pathlib import Path

class DualModelTester:
    """Test both Decision Tree and Random Forest models"""
    
    def __init__(self, dt_model_path='decision_tree_model.pkl', rf_model_path='random_forest_model.pkl'):
        """Initialize tester with both models"""
        print("📦 Loading models...")
        try:
            with open(dt_model_path, 'rb') as f:
                self.dt_model = pickle.load(f)
                print(f"✅ Decision Tree model loaded: {dt_model_path}")
        except:
            print(f"⚠️  Decision Tree model not found: {dt_model_path}")
            self.dt_model = None
        
        try:
            with open(rf_model_path, 'rb') as f:
                self.rf_model = pickle.load(f)
                print(f"✅ Random Forest model loaded: {rf_model_path}")
        except:
            print(f"⚠️  Random Forest model not found: {rf_model_path}")
            self.rf_model = None
    
    def test_on_dataset(self, X_test, y_test):
        """Test both models on a test dataset"""
        results = {}
        
        # Test Decision Tree
        if self.dt_model:
            print("\n🌳 Testing Decision Tree...")
            dt_predictions = self.dt_model.predict(X_test)
            
            dt_results = {
                'accuracy': accuracy_score(y_test, dt_predictions),
                'precision': precision_score(y_test, dt_predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_test, dt_predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, dt_predictions, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, dt_predictions)
            }
            results['decision_tree'] = dt_results
            print(f"   ✓ Accuracy: {dt_results['accuracy']:.4f}")
        
        # Test Random Forest
        if self.rf_model:
            print("\n🌲 Testing Random Forest...")
            rf_predictions = self.rf_model.predict(X_test)
            
            rf_results = {
                'accuracy': accuracy_score(y_test, rf_predictions),
                'precision': precision_score(y_test, rf_predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_test, rf_predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, rf_predictions, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, rf_predictions)
            }
            results['random_forest'] = rf_results
            print(f"   ✓ Accuracy: {rf_results['accuracy']:.4f}")
        
        return results
    
    def predict_with_both(self, features, model_type='both'):
        """
        Predict using one or both models
        model_type: 'decision_tree', 'random_forest', or 'both'
        """
        predictions = {}
        
        if model_type in ['decision_tree', 'both'] and self.dt_model:
            dt_pred = self.dt_model.predict(features)[0]
            dt_prob = max(self.dt_model.predict_proba(features)[0])
            predictions['decision_tree'] = {
                'prediction': dt_pred,
                'confidence': float(dt_prob)
            }
        
        if model_type in ['random_forest', 'both'] and self.rf_model:
            rf_pred = self.rf_model.predict(features)[0]
            rf_prob = max(self.rf_model.predict_proba(features)[0])
            predictions['random_forest'] = {
                'prediction': rf_pred,
                'confidence': float(rf_prob)
            }
        
        return predictions
    
    def test_on_images(self, image_folder, model_type='both'):
        """Test on a folder of images"""
        image_folder = Path(image_folder)
        results = []
        
        image_paths = list(image_folder.glob('**/*.jpg')) + list(image_folder.glob('**/*.png'))
        
        for img_path in image_paths:
            try:
                features = extract_features_from_image(str(img_path))
                predictions = self.predict_with_both(features, model_type)
                
                results.append({
                    'image': str(img_path),
                    'predictions': predictions
                })
            except:
                continue
        
        return results
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {}
        
        if self.dt_model:
            info['decision_tree'] = {
                'status': 'loaded',
                'tree_depth': self.dt_model.get_depth(),
                'n_leaves': self.dt_model.get_n_leaves()
            }
        else:
            info['decision_tree'] = {'status': 'not_loaded'}
        
        if self.rf_model:
            info['random_forest'] = {
                'status': 'loaded',
                'n_estimators': self.rf_model.n_estimators,
                'max_depth': self.rf_model.max_depth
            }
        else:
            info['random_forest'] = {'status': 'not_loaded'}
        
        return info


# Legacy - Single model tester for backward compatibility
class ModelTester:
    def __init__(self, model_path='random_forest_model.pkl'):
        """Initialize the tester with a trained model (uses Random Forest by default)"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def test_on_dataset(self, X_test, y_test):
        """Test model on a test dataset"""
        predictions = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_test, predictions)
        }
        
        return results
    
    def test_on_images(self, image_folder):
        """Test model on a folder of images"""
        image_folder = Path(image_folder)
        results = []
        
        for img_path in image_folder.glob('**/*.jpg'):
            try:
                features = extract_features_from_image(str(img_path))
                prediction = self.model.predict(features)[0]
                probability = self.model.predict_proba(features)[0].max()
                
                results.append({
                    'image': str(img_path),
                    'prediction': prediction,
                    'confidence': probability
                })
            except:
                continue
        
        return results
