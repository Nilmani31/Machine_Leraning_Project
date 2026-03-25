import os
import pickle
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from feature_extractor import extract_features_from_image
import cv2
from pathlib import Path

def extract_all_features(data_dir='training_data'):
    """
    Extract features from all images in the dataset.
    Returns: X (features), y (labels), class_names, class_mapping
    """
    
    X = []  # Features
    y = []  # Labels
    class_names = []
    class_mapping = {}
    
    print(f"\n{'='*70}")
    print(f"🔍 FEATURE EXTRACTION PHASE")
    print(f"{'='*70}")
    print(f"🔍 Scanning {data_dir} for disease folders...")
    
    # Scan all folders
    disease_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    
    if not disease_folders:
        print(f"❌ No folders found in {data_dir}")
        return None, None, None, None
    
    print(f"✅ Found {len(disease_folders)} disease classes:")
    for idx, disease in enumerate(disease_folders):
        print(f"   {idx}: {disease}")
        class_names.append(disease)
        class_mapping[disease] = idx
    
    # Extract features from all images
    image_count = 0
    for class_idx, disease_folder in enumerate(disease_folders):
        folder_path = os.path.join(data_dir, disease_folder)
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        
        print(f"\n📁 Processing {disease_folder} ({len(image_files)} images)...")
        
        failed = 0
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                features = extract_features_from_image(img_path)
                if features is not None:
                    features_flat = features.flatten() if hasattr(features, 'flatten') else features
                    if len(features_flat) > 0 and not np.isnan(features_flat).any():
                        X.append(features_flat)
                        y.append(class_idx)
                        image_count += 1
                        if image_count % 50 == 0:
                            print(f"   ✓ Processed {image_count} images...")
                    else:
                        failed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                continue
        
        print(f"   ✅ Completed: {len(image_files) - failed}/{len(image_files)} images")
    
    print(f"\n📊 Total images processed: {image_count}")
    
    if image_count == 0:
        print("❌ No valid images found.")
        return None, None, None, None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"🔧 Feature shape: {X.shape}")
    print(f"🔧 Labels shape: {y.shape}")
    
    return X, y, class_names, class_mapping


def train_decision_tree(X_train, X_test, y_train, y_test, class_names):
    """Train Decision Tree Classifier"""
    print(f"\n{'='*70}")
    print(f"🌳 DECISION TREE TRAINING")
    print(f"{'='*70}")
    
    print("🤖 Training Decision Tree model...")
    dt_model = DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    dt_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt_model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n✅ Decision Tree trained!")
    print(f"📈 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 Precision (weighted): {precision:.4f}")
    print(f"📊 Recall (weighted): {recall:.4f}")
    print(f"📊 F1-Score (weighted): {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    }
    
    return dt_model, metrics


def train_random_forest(X_train, X_test, y_train, y_test, class_names):
    """Train Random Forest Classifier"""
    print(f"\n{'='*70}")
    print(f"🌲 RANDOM FOREST TRAINING")
    print(f"{'='*70}")
    
    print("🤖 Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n✅ Random Forest trained!")
    print(f"📈 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 Precision (weighted): {precision:.4f}")
    print(f"📊 Recall (weighted): {recall:.4f}")
    print(f"📊 F1-Score (weighted): {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Feature importance
    feature_importance = rf_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    print(f"\n🎯 Top 10 Important Features:")
    for idx, fi in enumerate(top_features_idx):
        print(f"   {idx+1}. Feature {fi}: {feature_importance[fi]:.4f}")
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    }
    
    return rf_model, metrics


def save_models(dt_model, rf_model, class_names, class_mapping):
    """Save both models"""
    print(f"\n{'='*70}")
    print(f"💾 SAVING MODELS")
    print(f"{'='*70}")
    
    # Save Decision Tree
    dt_path = 'decision_tree_model.pkl'
    with open(dt_path, 'wb') as f:
        pickle.dump(dt_model, f)
    print(f"✅ Decision Tree saved: {dt_path}")
    
    # Save Random Forest
    rf_path = 'random_forest_model.pkl'
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"✅ Random Forest saved: {rf_path}")
    
    # Save class mapping
    mapping_file = 'class_mapping.json'
    mapping_data = {
        'class_names': class_names,
        'class_mapping': class_mapping
    }
    with open(mapping_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    print(f"✅ Class mapping saved: {mapping_file}")


def generate_comparison_report(dt_metrics, rf_metrics, class_names):
    """Generate comparison report"""
    print(f"\n{'='*70}")
    print(f"📊 MODELS COMPARISON REPORT")
    print(f"{'='*70}\n")
    
    comparison = {
        'Decision Tree': dt_metrics,
        'Random Forest': rf_metrics,
        'Comparison': {
            'Accuracy Difference': abs(dt_metrics['accuracy'] - rf_metrics['accuracy']),
            'Precision Difference': abs(dt_metrics['precision'] - rf_metrics['precision']),
            'Recall Difference': abs(dt_metrics['recall'] - rf_metrics['recall']),
            'F1-Score Difference': abs(dt_metrics['f1_score'] - rf_metrics['f1_score']),
            'Better Model': 'Random Forest' if rf_metrics['f1_score'] > dt_metrics['f1_score'] else 'Decision Tree'
        }
    }
    
    # Print comparison
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    ALGORITHM COMPARISON RESULTS                      │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│ Metric            │ Decision Tree     │ Random Forest    │ Winner    │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    
    # Accuracy
    dt_acc = f"{dt_metrics['accuracy']*100:.2f}%"
    rf_acc = f"{rf_metrics['accuracy']*100:.2f}%"
    acc_winner = "RF ✓" if rf_metrics['accuracy'] > dt_metrics['accuracy'] else "DT ✓"
    print(f"│ Accuracy          │ {dt_acc:17} │ {rf_acc:16} │ {acc_winner:9} │")
    
    # Precision
    dt_prec = f"{dt_metrics['precision']:.4f}"
    rf_prec = f"{rf_metrics['precision']:.4f}"
    prec_winner = "RF ✓" if rf_metrics['precision'] > dt_metrics['precision'] else "DT ✓"
    print(f"│ Precision         │ {dt_prec:17} │ {rf_prec:16} │ {prec_winner:9} │")
    
    # Recall
    dt_rec = f"{dt_metrics['recall']:.4f}"
    rf_rec = f"{rf_metrics['recall']:.4f}"
    rec_winner = "RF ✓" if rf_metrics['recall'] > dt_metrics['recall'] else "DT ✓"
    print(f"│ Recall            │ {dt_rec:17} │ {rf_rec:16} │ {rec_winner:9} │")
    
    # F1-Score
    dt_f1 = f"{dt_metrics['f1_score']:.4f}"
    rf_f1 = f"{rf_metrics['f1_score']:.4f}"
    f1_winner = "RF ✓" if rf_metrics['f1_score'] > dt_metrics['f1_score'] else "DT ✓"
    print(f"│ F1-Score          │ {dt_f1:17} │ {rf_f1:16} │ {f1_winner:9} │")
    
    print("└─────────────────────────────────────────────────────────────────────┘")
    
    # Recommendation
    print(f"\n🏆 RECOMMENDATION: {comparison['Comparison']['Better Model']} performs better overall!")
    
    # Save comparison report
    report_file = 'model_comparison_report.json'
    with open(report_file, 'w') as f:
        json.dump({
            'Decision Tree': {
                'accuracy': comparison['Decision Tree']['accuracy'],
                'precision': comparison['Decision Tree']['precision'],
                'recall': comparison['Decision Tree']['recall'],
                'f1_score': comparison['Decision Tree']['f1_score']
            },
            'Random Forest': {
                'accuracy': comparison['Random Forest']['accuracy'],
                'precision': comparison['Random Forest']['precision'],
                'recall': comparison['Random Forest']['recall'],
                'f1_score': comparison['Random Forest']['f1_score']
            },
            'Better Model': comparison['Comparison']['Better Model']
        }, f, indent=2)
    
    print(f"\n✅ Comparison report saved: {report_file}")
    
    return comparison


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🚀 DUAL MODEL TRAINING PIPELINE")
    print("   Training both Decision Tree and Random Forest classifiers")
    print("="*70)
    
    # Step 1: Extract features
    X, y, class_names, class_mapping = extract_all_features('training_data')
    
    if X is None:
        print("❌ Failed to extract features. Exiting.")
        return
    
    # Step 2: Split data
    print(f"\n{'='*70}")
    print(f"📚 DATA SPLITTING")
    print(f"{'='*70}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(y)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(y)*100:.1f}%)")
    
    # Step 3: Train both models
    dt_model, dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test, class_names)
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test, class_names)
    
    # Step 4: Save both models
    save_models(dt_model, rf_model, class_names, class_mapping)
    
    # Step 5: Generate comparison report
    comparison = generate_comparison_report(dt_metrics, rf_metrics, class_names)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Files created:")
    print(f"   ✓ decision_tree_model.pkl")
    print(f"   ✓ random_forest_model.pkl")
    print(f"   ✓ class_mapping.json")
    print(f"   ✓ model_comparison_report.json")
    print(f"\n📊 Results:")
    print(f"   → Decision Tree Accuracy: {dt_metrics['accuracy']*100:.2f}%")
    print(f"   → Random Forest Accuracy: {rf_metrics['accuracy']*100:.2f}%")
    print(f"   → Better Model: {comparison['Comparison']['Better Model']}")
    print(f"\n🚀 Next step: Choose a model in ui_app.py or run model_comparison_report.json")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
