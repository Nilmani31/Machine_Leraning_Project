import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from feature_extractor import extract_features_from_image
from pathlib import Path

class ModelTester:
    def __init__(self, model_path='crop_model.pkl'):
        """Initialize the tester with a trained model"""
        with open(model_path, 'rb') as f:
            obj = pickle.load(f)
            self.model = obj['model']
            self.le = obj['label_encoder']
    
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
                    'prediction': self.le.inverse_transform([prediction])[0],
                    'confidence': probability
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return pd.DataFrame(results)
    
    def get_model_info(self):
        """Get information about the model"""
        return {
            'model_type': type(self.model).__name__,
            'num_classes': len(self.le.classes_),
            'classes': list(self.le.classes_),
            'num_features': self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 'Unknown'
        }

def run_validation_tests():
    """Run comprehensive validation tests"""
    tester = ModelTester()
    
    tests_passed = 0
    tests_total = 3
    
    # Test 1: Model loads correctly
    try:
        info = tester.get_model_info()
        print("✓ Test 1 PASSED: Model loaded successfully")
        print(f"  - Model Type: {info['model_type']}")
        print(f"  - Classes: {info['num_classes']}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
    
    # Test 2: Model can make predictions
    try:
        dummy_features = np.random.rand(1, 27)
        prediction = tester.model.predict(dummy_features)
        print("✓ Test 2 PASSED: Model can make predictions")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
    
    # Test 3: Probabilities sum to 1
    try:
        dummy_features = np.random.rand(1, 27)
        proba = tester.model.predict_proba(dummy_features)[0]
        assert np.isclose(proba.sum(), 1.0), "Probabilities don't sum to 1"
        print("✓ Test 3 PASSED: Probabilities are valid")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
    
    print(f"\n{tests_passed}/{tests_total} tests passed")
    return tests_passed == tests_total
