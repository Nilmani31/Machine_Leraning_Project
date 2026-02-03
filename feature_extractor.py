import cv2
import numpy as np
from PIL import Image
import io

def extract_features_from_image(image_input):
    """
    Extract 27 features from a plant image
    Supports PIL Image, numpy array, or file path
    """
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image_input, Image.Image):
        img = np.array(image_input)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        img = cv2.imread(image_input)
    
    if img is None:
        raise ValueError("Could not load image")
    
    # Convert to different color spaces for feature extraction
    if len(img.shape) == 3 and img.shape[2] == 3:
        # BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    else:
        img_rgb = img
    
    # Convert to LAB color space
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB) if len(img_rgb.shape) == 3 else img_rgb
    
    # Convert to HSV color space
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV) if len(img_rgb.shape) == 3 else img_rgb
    
    features = []
    
    # Extract color statistics from RGB
    for channel in range(3):
        channel_data = img_rgb[:,:,channel].flatten()
        features.append(np.mean(channel_data))
        features.append(np.std(channel_data))
        features.append(np.min(channel_data))
    
    # Extract color statistics from LAB
    for channel in range(3):
        channel_data = img_lab[:,:,channel].flatten()
        features.append(np.mean(channel_data))
        features.append(np.std(channel_data))
    
    # Extract color statistics from HSV
    for channel in range(3):
        channel_data = img_hsv[:,:,channel].flatten()
        features.append(np.mean(channel_data))
        features.append(np.std(channel_data))
    
    # Texture features using Sobel edges
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    
    features.append(np.mean(sobelx))
    features.append(np.std(sobelx))
    features.append(np.mean(sobely))
    features.append(np.std(sobely))
    
    # Laplacian features
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(np.mean(laplacian))
    features.append(np.std(laplacian))
    
    # Histogram features
    hist_b = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
    
    features.append(np.mean(hist_b))
    features.append(np.mean(hist_g))
    features.append(np.mean(hist_r))
    
    # Ensure we have exactly 27 features
    features = np.array(features[:27])
    
    # Pad with zeros if needed
    if len(features) < 27:
        features = np.pad(features, (0, 27 - len(features)), mode='constant')
    
    return features.reshape(1, -1)

def validate_image(image_file):
    """
    Validate if the uploaded file is a valid image
    """
    try:
        img = Image.open(image_file)
        img.verify()
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"
