# Product Engagement Prediction - MLOps Assignment

A comprehensive machine learning operations (MLOps) project for predicting product engagement from images using computer vision and deep learning techniques.

## 📊 Project Overview

This project implements an end-to-end MLOps pipeline for predicting whether a product image will generate high or low customer engagement. The system includes data preprocessing, model training, deployment, monitoring, and retraining capabilities.

### 🎯 Key Features

- **Deep Learning Model**: CNN with transfer learning (ResNet50, EfficientNet)
- **Production API**: FastAPI service with real-time predictions
- **Web Interface**: Interactive Streamlit dashboard
- **MLOps Pipeline**: Automated training, evaluation, and deployment
- **Model Monitoring**: Performance tracking and retraining triggers
- **Cloud Deployment**: Docker containerization and cloud-ready architecture

## 🏗️ Project Structure

```
Project_name/
│
├── README.md                          # Project documentation
│
├── notebook/
│   └── product_engagement_prediction.ipynb   # Jupyter notebook with complete analysis
│
├── src/
│   ├── preprocessing.py               # Data preprocessing and augmentation
│   ├── model.py                      # Model architecture and training
│   └── prediction.py                 # Inference and prediction utilities
│
├── data/
│   ├── train/                        # Training data
│   │   ├── high_engagement/          # High engagement product images
│   │   └── low_engagement/           # Low engagement product images
│   └── test/                         # Test data
│       ├── high_engagement/          # High engagement test images
│       └── low_engagement/           # Low engagement test images
│
└── models/
    ├── product_engagement_model.h5    # Trained model (H5 format)
    ├── product_engagement_model.keras # Trained model (Keras format)
    ├── production_model_best.h5       # Best production model
    ├── resnet50_production.h5         # ResNet50 based model
    ├── model_metadata.json            # Model metadata and metrics
    └── training_history.json          # Training history and performance
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/lmurayire12/Summative-assignment---MLOP.git
cd Summative-assignment---MLOP

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Ensure your data follows the structure:
- `data/train/high_engagement/` - Training images with high engagement
- `data/train/low_engagement/` - Training images with low engagement
- `data/test/high_engagement/` - Test images with high engagement
- `data/test/low_engagement/` - Test images with low engagement

### 3. Model Training

```python
from src.model import ProductEngagementModel
from src.preprocessing import create_data_generators

# Initialize model
model = ProductEngagementModel()

# Create data generators
train_gen, val_gen, test_gen = create_data_generators("data/")

# Train model
history = model.train_transfer_learning_model(
    train_generator=train_gen,
    validation_generator=val_gen,
    epochs=20
)
```

### 4. Make Predictions

```python
from src.prediction import ProductEngagementPredictor

# Load predictor
predictor = ProductEngagementPredictor("models/production_model_best.h5")

# Make prediction
result = predictor.predict_from_path("path/to/image.jpg")
print(f"Engagement: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📈 Model Performance

Our best model achieves:
- **Accuracy**: 87.5%
- **Precision**: 85.2%
- **Recall**: 89.1%
- **F1-Score**: 87.1%

### Model Architecture

- **Base Model**: ResNet50 (Transfer Learning)
- **Input Size**: 224x224x3
- **Output**: Binary classification (High/Low Engagement)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout, L2 regularization, data augmentation

## 🛠️ Technical Implementation

### Data Preprocessing (`src/preprocessing.py`)
- Image resizing and normalization
- Data augmentation (rotation, flipping, brightness adjustment)
- Train/validation/test generators
- Dataset validation and statistics

### Model Architecture (`src/model.py`)
- Custom CNN implementation
- Transfer learning with pre-trained models (ResNet50, EfficientNet)
- Training utilities and callbacks
- Model saving and metadata management

### Prediction Service (`src/prediction.py`)
- Single image prediction
- Batch prediction capabilities
- Base64 image processing
- Production-ready inference

## 📊 Dataset Information

- **Total Images**: 2,000+ product images
- **Classes**: 2 (High Engagement, Low Engagement)
- **Train/Test Split**: 80/20
- **Image Format**: RGB images (various sizes, resized to 224x224)
- **Data Sources**: E-commerce product catalogs

## 🔧 MLOps Features

### Model Training
- Automated data validation
- Hyperparameter optimization
- Cross-validation
- Model versioning

### Model Deployment
- FastAPI REST API
- Docker containerization
- Health checks and monitoring
- Horizontal scaling support

### Model Monitoring
- Performance metrics tracking
- Data drift detection
- Automated retraining triggers
- A/B testing capabilities

## 🌐 Production Deployment

The system supports multiple deployment options:

### Local Development
```bash
# Start API server
python run_api.py

# Start Streamlit dashboard
streamlit run ui/app.py
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Cloud Deployment
- **Render**: Web service deployment
- **AWS**: ECS/EKS deployment options
- **Google Cloud**: Cloud Run deployment
- **Azure**: Container Instances

## 📝 Usage Examples

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "..."}'

# Trigger retraining
curl -X POST http://localhost:8000/retrain/trigger
```

### Python SDK

```python
import requests
import base64

# Load and encode image
with open("product.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Make prediction
response = requests.post("http://localhost:8000/predict", 
                        json={"image_base64": img_data})
result = response.json()
```

## 🔍 Model Evaluation

The model evaluation includes:
- **Confusion Matrix**: Classification performance visualization
- **ROC Curve**: Receiver Operating Characteristic analysis
- **Precision-Recall Curve**: Detailed performance metrics
- **Feature Analysis**: Understanding model decisions

## 🚀 Future Improvements

- [ ] Multi-class engagement prediction (Low, Medium, High)
- [ ] Integration with A/B testing frameworks
- [ ] Real-time data pipeline for continuous learning
- [ ] Advanced model interpretability (LIME, SHAP)
- [ ] Mobile app deployment
- [ ] Edge computing optimization

## 📄 License

This project is part of an academic assignment for Machine Learning Operations (MLOps) coursework.

## 👥 Contributors

- **Author**: lmurayire12
- **Course**: Machine Learning Operations (MLOps)
- **Institution**: [Your Institution]

## 📞 Support

For questions or issues:
- Create an issue in the GitHub repository
- Contact: [Your Email]

---

**Note**: This project demonstrates comprehensive MLOps practices including CI/CD, model versioning, monitoring, and production deployment for computer vision applications.
