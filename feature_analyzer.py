"""
Analyze training data features to provide disease-specific recommendations
"""
import numpy as np
import os
from pathlib import Path
from feature_extractor import extract_features_from_image
from class_mapping import DISEASE_MAPPING

def get_disease_feature_stats():
    """
    Calculate mean and std dev of each feature for each disease
    Returns dict: {disease_name: {'mean': [...], 'std': [...], 'min': [...], 'max': [...]}}
    """
    training_dir = "training_data"
    disease_stats = {}
    
    if not os.path.exists(training_dir):
        return disease_stats
    
    # Process each disease folder
    for disease_folder in os.listdir(training_dir):
        folder_path = os.path.join(training_dir, disease_folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        features_list = []
        image_count = 0
        
        # Extract features from all images in this disease folder
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, img_file)
                try:
                    features = extract_features_from_image(img_path)
                    if features is not None:
                        features_list.append(features.flatten())
                        image_count += 1
                except:
                    continue
        
        # Calculate statistics if we found images
        if features_list:
            features_array = np.array(features_list)
            disease_stats[disease_folder] = {
                'mean': features_array.mean(axis=0).tolist(),
                'std': features_array.std(axis=0).tolist(),
                'min': features_array.min(axis=0).tolist(),
                'max': features_array.max(axis=0).tolist(),
                'sample_count': image_count
            }
    
    return disease_stats

def get_disease_feature_means():
    """
    Get the mean feature values for each disease (for quick reference)
    Returns: {disease_name: [27 feature means]}
    """
    stats = get_disease_feature_stats()
    return {disease: data['mean'] for disease, data in stats.items()}

def get_feature_range_for_disease(disease_name):
    """
    Get min and max feature values for a specific disease
    Returns: {feature_index: {'min': value, 'max': value, 'mean': value}}
    """
    stats = get_disease_feature_stats()
    
    if disease_name not in stats:
        return None
    
    disease_data = stats[disease_name]
    feature_ranges = {}
    
    for i in range(27):
        feature_ranges[i] = {
            'min': disease_data['min'][i],
            'max': disease_data['max'][i],
            'mean': disease_data['mean'][i],
            'std': disease_data['std'][i]
        }
    
    return feature_ranges
