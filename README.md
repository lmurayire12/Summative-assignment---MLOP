Product Engagement Predictor - Complete MLOps System
A production-ready machine learning system for predicting customer engagement from product images using deep learning. Features complete MLOps pipeline with FastAPI backend, Streamlit dashboard, and cloud deployment.

## 🌐 Live Demo & Links

🎥 **Live Demo Video:**  https://youtu.be/grZw7ItXP2k
🚀 **Live API Backend:** [https://product-engagement-predictor.onrender.com](https://product-engagement-predictor.onrender.com)    
📚 **API Documentation:** (https://product-engagement-predictor.onrender.com/docs)  
📱 **GitHub Repository:** (https://github.com/lmurayire12/Summative-assignment---MLOP)

---

## 🔧 Features & Capabilities

✅ **Deep Learning Model:** CNN + ResNet50 transfer learning achieving 87.5% accuracy  
✅ **Real-time Predictions:** Upload images and get instant engagement predictions  
✅ **Production API:** FastAPI with comprehensive endpoints and monitoring  
✅ **Interactive Dashboard:** Streamlit web interface for easy access  
✅ **Model Optimization:** Compressed from 103MB to 0.56MB (99.5% reduction!)  
✅ **Load Testing:** Performance validated with Locust testing framework  
✅ **Cloud Deployment:** Live on Render with automatic scaling  
✅ **MLOps Pipeline:** Complete training, evaluation, and deployment automation  
✅ **Monitoring & Logging:** Real-time performance tracking and health checks

## 📊 System Performance

- **🎯 Model Accuracy:** 87.5%
- **📦 Model Size:** 0.56 MB (optimized)
- **⚡ Response Time:** <2.4 seconds average
- **🚀 Success Rate:** 100% (load tested)
- **👥 Concurrent Users:** 5+ users tested
- **📈 Throughput:** 0.5 requests/second

---

## 🚀 Quick Start Guide

### 📋 Prerequisites

```bash
# Required software
- Python 3.8+ 
- pip (Python package manager)
- Git

# Check your Python version
python --version
```

### 1️⃣ Clone & Setup

```bash
# Clone the repository
git clone https://github.com/lmurayire12/Summative-assignment---MLOP.git
cd Summative-assignment---MLOP

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run the FastAPI Backend

```bash
# Navigate to API directory
cd api

# Method 1: Run with Python
python main.py

# Method 2: Run with Uvicorn (production-like)
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Method 3: Run with custom settings
python main.py --host 0.0.0.0 --port 8000 --reload
```

**🔗 API will be available at:**
- Main API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 3️⃣ Run the Streamlit Dashboard

```bash
# Open new terminal/command prompt
cd ui

# Run the main dashboard
streamlit run app.py

# Alternative: Run specific version
streamlit run app_final.py
streamlit run app_enhanced.py
```

**🔗 Dashboard will be available at:**
- Main Dashboard: http://localhost:8501
- Network Access: http://YOUR_IP:8501

### 4️⃣ Test the System

```bash
# Test API health
curl http://localhost:8000/health

# Test prediction (if API is running)
# Upload an image via the Streamlit dashboard
# Or use the Swagger UI at http://localhost:8000/docs
```


## 🔌 API Endpoints Reference

### 🏥 Health & Monitoring

| Endpoint | Method | Description | Example |
|----------|--------|-------------|---------|
| `/` | GET | Basic health check | `curl http://localhost:8000/` |
| `/health` | GET | Detailed system health | `curl http://localhost:8000/health` |
| `/model/info` | GET | Model metadata & stats | `curl http://localhost:8000/model/info` |
| `/metrics` | GET | Performance metrics | `curl http://localhost:8000/metrics` |

### 🔮 Prediction Endpoints

| Endpoint | Method | Description | Usage |
|----------|--------|-------------|-------|
| `/predict` | POST | JSON prediction (base64) | Upload base64 encoded image |
| `/predict/file` | POST | File upload prediction | Upload image file directly |
| `/predict/batch` | POST | Batch predictions | Multiple images at once |

### 🔄 Model Management

| Endpoint | Method | Description | Purpose |
|----------|--------|-------------|---------|
| `/retrain/trigger` | POST | Trigger retraining | Model updates |
| `/monitoring/stats` | GET | Performance statistics | System monitoring |
| `/monitoring/predictions` | GET | Recent predictions | Activity logs |

---

## 📄 License & Attribution

This project is part of an MLOps assignment demonstrating complete machine learning pipeline implementation from training to production deployment.

**Technologies Used:**
- 🐍 **Python** - Core programming language
- 🚀 **FastAPI** - Modern web framework for APIs
- 🎨 **Streamlit** - Interactive web applications
- 🧠 **TensorFlow** - Machine learning framework
- 🌐 **Render** - Cloud deployment platform
- 🧪 **Locust** - Load testing framework

---


## 🏆 Project Achievements

✅ **Complete MLOps Pipeline** - Training to production deployment  
✅ **Production-Ready API** - Comprehensive endpoints with monitoring  
✅ **User-Friendly Interface** - Interactive Streamlit dashboard  
✅ **Model Optimization** - 99.5% size reduction while maintaining accuracy  
✅ **Performance Validated** - Load tested with 100% success rate  
✅ **Cloud Deployed** - Live system with automatic scaling  
✅ **Business Value** - Real-world applications with measurable ROI  
✅ **Professional Documentation** - Complete setup and usage guides

---
