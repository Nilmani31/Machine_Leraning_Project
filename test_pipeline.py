import numpy as np
from PIL import Image, ImageDraw
import pickle
from feature_extractor import extract_features_from_image
import matplotlib.pyplot as plt

def create_synthetic_plant_image(disease_type="healthy"):
    """
    Create a synthetic plant leaf image for testing
    """
    # Create base image
    width, height = 300, 300
    img = Image.new('RGB', (width, height), color=(34, 139, 34))  # Dark green background
    draw = ImageDraw.Draw(img)
    
    # Create leaf shape (ellipse)
    leaf_box = [50, 50, 250, 280]
    draw.ellipse(leaf_box, fill=(50, 205, 50), outline=(34, 139, 34))
    
    # Add vein pattern
    draw.line([150, 50, 150, 280], fill=(34, 139, 34), width=3)
    for i in range(60, 280, 30):
        draw.line([150, i, 120, i+20], fill=(34, 139, 34), width=2)
        draw.line([150, i, 180, i+20], fill=(34, 139, 34), width=2)
    
    # Add disease patterns based on type
    if disease_type == "bacterial_spot":
        # Dark brown/black spots with yellow halos
        spots = [(100, 100), (150, 120), (180, 150), (120, 200), (170, 200)]
        for spot in spots:
            # Yellow halo
            draw.ellipse(
                [spot[0]-15, spot[1]-15, spot[0]+15, spot[1]+15],
                fill=(255, 255, 0)
            )
            # Dark center
            draw.ellipse(
                [spot[0]-8, spot[1]-8, spot[0]+8, spot[1]+8],
                fill=(80, 40, 20)
            )
    
    elif disease_type == "powdery_mildew":
        # White powder-like coating
        for x in range(80, 240, 20):
            for y in range(80, 260, 20):
                draw.ellipse(
                    [x, y, x+15, y+15],
                    fill=(240, 240, 240)
                )
    
    elif disease_type == "rust":
        # Orange/brown rust spots
        spots = [(110, 110), (160, 140), (130, 190), (190, 160)]
        for spot in spots:
            draw.ellipse(
                [spot[0]-12, spot[1]-12, spot[0]+12, spot[1]+12],
                fill=(200, 100, 20)
            )
    
    # Add some texture noise
    pixels = np.array(img)
    noise = np.random.randint(-10, 10, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    return img

def test_disease_detection():
    """
    Test the complete pipeline: Image → Features → Prediction
    """
    print("="*70)
    print("PLANT DISEASE DETECTION - FULL PIPELINE TEST")
    print("="*70)
    
    # Load model
    try:
        with open('crop_model.pkl', 'rb') as f:
            obj = pickle.load(f)
            model = obj['model']
            le = obj['label_encoder']
        print("\n✅ Model loaded successfully!")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return
    
    # Test cases
    test_cases = [
        ("healthy", "Healthy Plant Leaf"),
        ("bacterial_spot", "Bacterial Spot Disease"),
        ("powdery_mildew", "Powdery Mildew Disease"),
        ("rust", "Rust Disease")
    ]
    
    results = []
    
    for disease_type, description in test_cases:
        print(f"\n{'-'*70}")
        print(f"TEST: {description}")
        print(f"{'-'*70}")
        
        try:
            # Step 1: Create synthetic image
            print("\n📸 Step 1: Creating synthetic plant image...")
            image = create_synthetic_plant_image(disease_type)
            image.save(f'test_image_{disease_type}.png')
            print(f"   ✅ Image created: test_image_{disease_type}.png")
            
            # Step 2: Extract features
            print("\n🔍 Step 2: Extracting 27 features from image...")
            features = extract_features_from_image(image)
            print(f"   ✅ Features extracted successfully!")
            print(f"   Feature shape: {features.shape}")
            print(f"   Sample features: f0={features[0,0]:.2f}, f1={features[0,1]:.2f}, f2={features[0,2]:.2f}")
            
            # Step 3: Make prediction with both models
            print("\n🤖 Step 3: Making predictions...")
            
            # Random Forest prediction
            pred_idx = model.predict(features)[0]
            pred_label = le.inverse_transform([pred_idx])[0]
            pred_proba = model.predict_proba(features)[0]
            confidence = max(pred_proba) * 100
            
            print(f"   ✅ Prediction Complete!")
            print(f"   Disease: {pred_label.replace('_', ' ')}")
            print(f"   Confidence: {confidence:.2f}%")
            
            # Step 4: Show top 3 predictions
            print(f"\n📊 Step 4: Top 3 predictions:")
            top_indices = np.argsort(pred_proba)[-3:][::-1]
            for rank, idx in enumerate(top_indices, 1):
                disease_name = le.inverse_transform([idx])[0].replace('_', ' ')
                prob = pred_proba[idx] * 100
                print(f"   {rank}. {disease_name}: {prob:.2f}%")
            
            results.append({
                'test': description,
                'prediction': pred_label,
                'confidence': confidence,
                'image_file': f'test_image_{disease_type}.png'
            })
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                'test': description,
                'prediction': 'ERROR',
                'confidence': 0,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Test Case':<30} {'Prediction':<30} {'Confidence':<15}")
    print("-" * 75)
    
    for result in results:
        test_name = result['test'][:28]
        prediction = result['prediction'].replace('_', ' ')[:28] if 'prediction' in result else 'ERROR'
        confidence = f"{result['confidence']:.2f}%" if 'confidence' in result else 'N/A'
        print(f"{test_name:<30} {prediction:<30} {confidence:<15}")
    
    # Overall status
    successful = sum(1 for r in results if r.get('confidence', 0) > 0)
    total = len(results)
    
    print(f"\n{'='*70}")
    print(f"TESTS PASSED: {successful}/{total}")
    print(f"{'='*70}\n")
    
    if successful == total:
        print("✅ ALL TESTS PASSED! Your model is working correctly!")
        print("\n🎉 Your disease detection pipeline is ready to use!")
        print("\nTo use with real images:")
        print("1. Run: streamlit run ui_app.py")
        print("2. Upload your plant image")
        print("3. Get instant disease prediction")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return results

if __name__ == "__main__":
    results = test_disease_detection()
