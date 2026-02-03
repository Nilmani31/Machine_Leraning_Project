# Plant Disease Detection System

## Project Overview
This project implements a machine learning system to detect and classify plant diseases from image features using Random Forest and Decision Tree classifiers.

## 📊 Dataset Information
- **Source**: PlantVillage Dataset
- **Image Type**: Plant leaf images with various diseases
- **Features**: 27 extracted numerical features per image
- **Feature Extraction**: Color statistics (RGB, LAB, HSV), texture features (Sobel, Laplacian), and histogram features

### Sample Features (f0-f26):
- `f0-f2`: RGB color means
- `f3-f5`: RGB color standard deviations
- `f6-f8`: LAB color means
- `f9-f10`: LAB color standard deviations
- `f11-f14`: HSV color statistics
- `f15-f22`: Edge and texture features (Sobel, Laplacian)
- `f23-f26`: Histogram features

## 🏗️ Model Architecture

### Models Used:
1. **Random Forest Classifier**
   - Ensemble of decision trees
   - Provides probability estimates
   - Better generalization

2. **Decision Tree Classifier**
   - Single tree model
   - Interpretable results
   - Used as baseline

### Model Parameters:
```python
Random Forest:
- n_estimators: Default (100)
- max_depth: Auto
- random_state: Fixed for reproducibility

Decision Tree:
- max_depth: Auto
- random_state: Fixed for reproducibility
```

## 📈 Model Performance

### Key Metrics:
- **Accuracy**: 85%+ (on test data)
- **Precision**: 84%+ (weighted average)
- **Recall**: 85%+ (weighted average)
- **F1-Score**: 84%+ (weighted average)

### Performance by Disease:
See `model_metrics.json` for detailed per-class metrics

## 🛠️ Project Structure

```
App/
├── ui_app.py              # Main Streamlit application
├── crop_model.pkl         # Trained model and label encoder
├── evaluation.py          # Model evaluation metrics
├── feature_extractor.py   # Feature extraction from images
├── test_model.py          # Testing and validation
├── model_metrics.json     # Saved evaluation metrics
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🚀 Installation & Setup

### Prerequisites:
- Python 3.8+
- pip package manager

### Installation Steps:

1. **Clone/Download the project**
   ```bash
   cd "c:\Users\chamsha nilmani\Documents\semester 5\ML\App"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run ui_app.py
   ```

4. **Access the app**
   - Open browser to `http://localhost:8501`

## 📋 Features & Usage

### Two Input Methods:

#### 1. Image Upload
- Upload plant leaf images (JPG, PNG)
- Automatic feature extraction
- Direct disease prediction

#### 2. Manual Feature Input
- Adjust 27 features using sliders
- Get instant predictions
- View confidence scores

### Output Information:
- **Disease Prediction**: Classified disease name
- **Confidence Score**: Prediction probability (0-100%)
- **Disease Info**: Symptoms and treatment recommendations
- **Probability Table**: All diseases with probabilities

## 🧪 Testing

Run validation tests:
```bash
python test_model.py
```

Tests verify:
- Model loading
- Prediction capability
- Probability validity
- Feature extraction
- Image validation

## 📊 Evaluation Metrics

Load and view saved metrics:
```python
from evaluation import load_evaluation_metrics
metrics = load_evaluation_metrics()
print(f"Accuracy: {metrics['accuracy']}")
print(f"F1-Score: {metrics['f1_score']}")
```

## 🌐 Deployment Options

### Local Deployment (Current):
```bash
streamlit run ui_app.py
```

### Cloud Deployment:

#### 1. Streamlit Cloud
```bash
# Push to GitHub
git push origin main

# Deploy via https://share.streamlit.io
```

#### 2. Heroku
```bash
# Create Procfile
echo "web: streamlit run ui_app.py" > Procfile

# Deploy
heroku create
git push heroku main
```

#### 3. AWS EC2
```bash
# SSH to instance
ssh -i key.pem ec2-user@instance-ip

# Install and run
pip install -r requirements.txt
streamlit run ui_app.py --server.port 80
```

#### 4. Docker Containerization
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "ui_app.py"]
```

## 📝 Model Training

To retrain the model:

```python
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your training data
data = pd.read_csv('features.csv')
X = data.iloc[:, 2:]  # Features (f0-f26)
y = data['label']     # Labels

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_encoded)

# Save model
model_dict = {'model': rf_model, 'label_encoder': le}
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model_dict, f)
```

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "ValueError: X has 3 features, but RandomForestClassifier is expecting 27 features"
**Solution**: Ensure extracted features have exactly 27 dimensions

### Issue: Image upload not working
**Solution**: Check file format (JPG, PNG) and file size

## 📧 Support & Contact
For issues or questions, refer to the project proposal document: `EE7209_ProjectProposal_GP11_EG20225211_EG20225215.pdf`

## 📄 License
Academic Project - Semester 5 ML Course

## 👥 Project Contributors
- EG20225211
- EG20225215

---
**Last Updated**: November 2025
**Status**: Complete with UI, Evaluation, Testing, and Documentation
