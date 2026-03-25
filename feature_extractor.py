import cv2
import numpy as np

def extract_features_from_image(image_path):
    """
    Extract 27 optimized features from plant image
    """
    try:
        # Read image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                return None
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(image_path, dtype=np.uint8)

        # Resize for consistency
        img = cv2.resize(img, (224, 224))
        
        # Ensure correct dtype
        img = img.astype(np.uint8)

        features = []

        # ===== RGB Features (f0-f5) =====
        img_float = img.astype(np.float64) / 255.0
        rgb_mean = np.array([img_float[:,:,0].mean(), img_float[:,:,1].mean(), img_float[:,:,2].mean()])
        rgb_std = np.array([img_float[:,:,0].std(), img_float[:,:,1].std(), img_float[:,:,2].std()])
        features.extend(rgb_mean)
        features.extend(rgb_std)

        # ===== LAB Features (f6-f10) =====
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
        lab_mean = np.array([lab_img[:,:,0].mean(), lab_img[:,:,1].mean(), lab_img[:,:,2].mean()])
        lab_std = np.array([lab_img[:,:,0].std(), lab_img[:,:,1].std(), lab_img[:,:,2].std()])
        features.extend(lab_mean)
        features.extend(lab_std)

        # ===== HSV Features (f11-f14) =====
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float64)
        hsv_mean = np.array([hsv_img[:,:,0].mean(), hsv_img[:,:,1].mean(), hsv_img[:,:,2].mean()])
        features.extend(hsv_mean)

        # ===== Edge Detection (f15-f22) =====
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features.extend([sobelx.mean(), sobely.mean(), sobelx.std(), sobely.std()])

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([laplacian.mean(), laplacian.std()])
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([magnitude.mean(), magnitude.std()])

        # ===== Histogram Features (f23-f26) =====
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().mean()
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten().mean()
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten().mean()
        gray_hist = np.histogram(gray, bins=256)[0]
        hist_entropy = -np.sum((gray_hist / gray.size) * np.log2((gray_hist / gray.size) + 1e-10))
        features.extend([hist_r, hist_g, hist_b, hist_entropy])

        return np.array(features, dtype=np.float32).reshape(1, -1)

    except Exception as e:
        return None

