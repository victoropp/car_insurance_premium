# 🏢 Videbimus AI - Insurance Premium Analytics Platform

**World-Class Streamlit Dashboard for ML-Powered Insurance Premium Prediction**

Developed by: **Victor Collins Oppon**  
Company: **Videbimus AI**  
Website: https://www.videbimusai.com  
Contact: consulting@videbimusai.com

---

## 🌟 Features

- **18 Interactive Visualizations** across 4 comprehensive sections
- **99.78% ML Accuracy** with ensemble model predictions
- **Real-time Premium Calculator** using advanced ML models
- **Memory Optimized** for Streamlit Cloud deployment
- **Professional UI/UX** with responsive design
- **Production-Grade Code** with caching and error handling

## 📊 Dashboard Sections

### 1. 📊 Executive Summary
High-level business insights with 6 strategic visualizations:
- Premium distribution overview
- Model performance summary
- Age vs premium analysis
- Geographic risk distribution
- Experience impact analysis
- Feature importance ranking

### 2. 🔍 Detailed Analysis
Deep-dive analytics with 6 comprehensive visualizations:
- Experience level distribution
- Vehicle age impact
- Accident history analysis
- Annual mileage patterns
- Premium correlation matrix
- Top feature insights

### 3. 🎯 Model Performance
ML excellence dashboard with 6 advanced metrics:
- Top 10 models validation performance
- Ensemble models test set results
- Overfitting analysis
- Model rankings with distinct colors
- Performance metrics comparison
- Best model indicator gauge

### 4. 🧮 Premium Calculator
AI-powered real-time predictions featuring:
- Interactive input forms
- Multi-model ensemble predictions
- Risk assessment breakdown
- Premium optimization recommendations
- Professional results display

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Streamlit account (for deployment)
- GitHub account

### Local Development

1. **Clone and Setup**
   ```bash
   cd streamlit_dashboard
   pip install -r requirements.txt
   ```

2. **Run Locally**
   ```bash
   streamlit run app.py
   ```

3. **Access Dashboard**
   ```
   http://localhost:8501
   ```

### Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy Streamlit dashboard"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `streamlit_dashboard` folder
   - Set main file: `app.py`
   - Deploy!

---

## 📁 Project Structure

```
streamlit_dashboard/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── src/
│   └── streamlit_visualizations.py # Optimized visualization engine
├── data/                          # Dataset files
│   ├── insurance_tranining_dataset.csv
│   ├── final_test_results.csv
│   ├── model_results.csv
│   ├── feature_importance.csv
│   └── predictions_holdout_test.csv
├── models/                        # Trained ML models
│   ├── stacking_linear.pkl       # Best performing model
│   ├── stacking_ridge.pkl        # Ridge ensemble model
│   └── voting_ensemble.pkl       # Voting ensemble model
└── assets/                        # Static assets (if needed)
```

---

## 🔧 Technical Architecture

### Memory Optimization
- **@st.cache_resource**: Models loaded once and cached
- **@st.cache_data**: Data processed once and cached
- **Lazy Loading**: Models loaded on-demand for predictions
- **Garbage Collection**: Memory cleanup after operations
- **Apache Arrow**: Fast data serialization enabled

### Model Pipeline
- **Feature Engineering**: Automated preprocessing pipeline
- **Ensemble Methods**: 3 production-grade ML models
- **Robust Scaling**: Standardized feature scaling
- **Error Handling**: Comprehensive exception management

### UI/UX Design
- **Professional Branding**: Videbimus AI corporate identity
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Elements**: Plotly charts with hover details
- **Progress Indicators**: Loading spinners and status updates

---

## 📈 Model Performance

| Model | Test R² Score | Test RMSE | Test MAE |
|-------|---------------|-----------|----------|
| **Stacking (Linear)** | **0.9978** | 0.2721 | 0.2010 |
| Stacking (Ridge) | 0.9978 | 0.2725 | 0.2012 |
| Voting Ensemble | 0.9948 | 0.4190 | 0.2939 |

**Best Model**: Stacking (Linear) with 99.78% R² accuracy

---

## 🎯 Key Features

### ✅ Dashboard Capabilities
- [x] 18 interactive visualizations
- [x] Real-time ML predictions
- [x] Professional UI/UX design
- [x] Memory-optimized performance
- [x] Mobile-responsive layout
- [x] Error handling & validation
- [x] Loading states & progress indicators
- [x] Comprehensive tooltips & help text

### ✅ ML Pipeline
- [x] Feature engineering automation
- [x] Ensemble model predictions
- [x] Robust data preprocessing
- [x] Model performance validation
- [x] Risk assessment analytics
- [x] Premium optimization recommendations

### ✅ Production Features
- [x] Streamlit Cloud deployment ready
- [x] Environment configuration
- [x] Memory caching optimization
- [x] Error logging and monitoring
- [x] Professional branding
- [x] Contact integration

---

## 🔒 Data Privacy & Security

- **No Personal Data Storage**: All calculations performed client-side
- **Secure ML Models**: Pre-trained models with no data leakage
- **Privacy Compliant**: No user data retention or tracking
- **Professional Standards**: Enterprise-grade security practices

---

## 🤝 Support & Contact

### Technical Support
- **Email**: consulting@videbimusai.com
- **Website**: https://www.videbimusai.com

### Business Inquiries
- **Data Science Consulting**: ML model development and deployment
- **Custom Dashboard Development**: Tailored analytics solutions
- **Training & Workshops**: Streamlit and ML best practices

---

## 📜 License

© 2024 **Videbimus AI**. All rights reserved.

This dashboard is developed for portfolio and demonstration purposes. 
For commercial use or licensing, please contact consulting@videbimusai.com

---

## 🙏 Acknowledgments

- **Streamlit Team**: For the amazing framework
- **Plotly**: For interactive visualization capabilities
- **Scikit-learn**: For robust ML algorithms
- **Open Source Community**: For the foundation tools

---

**Built with ❤️ by Victor Collins Oppon @ Videbimus AI**