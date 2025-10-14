# 🎯 Allocation Maximizer

> **AI-Powered Supply Chain Allocation Optimization Platform**

[![Railway Deploy](https://img.shields.io/badge/Deploy%20on-Railway-0B0D0E?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)

A comprehensive full-stack application for optimizing supply chain allocation using advanced ML forecasting and intelligent distribution algorithms. Built with modern technologies and ready for production deployment.

## 🌟 Features

### 🎨 **Frontend Experience**
- **🔐 OptiU-Branded Authentication** - Secure login with company branding
- **📊 Interactive Dashboard** - Real-time allocation optimization interface  
- **🤖 ML-Powered Forecasting** - Multiple model selection (LSTM, Prophet, Random Forest, SARIMA, XGBoost)
- **📈 Dynamic Charts** - Interactive visualization with Chart.js
- **📱 Responsive Design** - Mobile-friendly interface with Tailwind CSS
- **🎯 Real-time Results** - Live allocation optimization with detailed analytics

### 🔧 **Backend Architecture**
- **⚡ FastAPI Framework** - High-performance async API
- **🧠 ML Model Integration** - Advanced forecasting algorithms
- **📊 CSV Data Processing** - Robust data validation and preprocessing
- **🎛️ Multiple Optimization Strategies**:
  - Fair Share Allocation
  - Priority-Based Distribution
- **🔄 Redis Caching** - Performance optimization
- **📋 Comprehensive Logging** - Full request/response tracking

### 🎯 **Business Intelligence**
- **📊 Real Business Names** - Customers, products, and distribution centers
- **🎪 Allocation Scenarios** - Multiple optimization strategies
- **📈 Performance Metrics** - Detailed analytics and reporting
- **🔍 Decision Factors** - Transparent allocation reasoning

## 🚀 Quick Start

### 🔑 **Demo Access**
```
Username: demo
Password: Demo@2024$SCX!
```

### 🏃‍♂️ **Local Development**

#### Prerequisites
- Python 3.11+
- Node.js 18+
- npm/yarn

#### 1️⃣ **Backend Setup**
```bash
cd backend/api/module4
pip install -r ../../../requirements.txt
python run_server.py
```
🌐 Backend runs on: `http://localhost:8000`

#### 2️⃣ **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```
🌐 Frontend runs on: `http://localhost:3000`

### 🚀 **Production Deployment (Railway)**

#### One-Click Deploy
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/nourdesoukizz/allocation-maximizer)

#### Manual Setup
1. **Fork/Clone** this repository
2. **Connect to Railway** at [railway.app](https://railway.app)
3. **Deploy** - Railway auto-detects configuration
4. **Access** your live application!

## 📁 Project Structure

```
allocation-maximizer/
├── 🎨 frontend/                 # React TypeScript Frontend
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── contexts/          # React contexts (Auth)
│   │   └── services/          # API integration
│   ├── public/               # Static assets
│   └── dist/                # Production build
├── 🔧 backend/                 # FastAPI Python Backend
│   └── api/module4/
│       ├── routers/          # API endpoints
│       ├── services/         # Business logic
│       ├── optimizers/       # Allocation algorithms
│       ├── ml_models/        # Machine learning models
│       └── utils/            # Helper functions
├── 📊 data/                   # CSV data files
├── 📚 docs/                   # Documentation
├── 🧪 tests/                  # Test suites
├── 🚀 main.py                # Production entry point
├── 📋 requirements.txt       # Python dependencies
└── 🛠️ railway.json          # Deployment configuration
```

## 🛠️ Technology Stack

### **Frontend**
- **React 18** - Modern UI framework
- **TypeScript** - Type-safe development
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **Chart.js** - Interactive charts
- **React Router** - Client-side routing

### **Backend**
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **PuLP** - Optimization algorithms
- **Redis** - Caching layer

### **Deployment**
- **Railway** - Cloud deployment platform
- **Nixpacks** - Automatic build detection
- **Docker** - Containerization ready

## 🎯 Usage Examples

### 🔍 **Basic Allocation**
1. **Login** with demo credentials
2. **Select** customers, distribution centers, and products
3. **Choose** optimization strategy (Fair Share/Priority)
4. **Run** allocation optimization
5. **Analyze** results with interactive charts

### 🤖 **ML Forecasting**
1. **Click** on any allocation result
2. **Select** ML model (LSTM, Prophet, etc.)
3. **View** historical vs forecasted data
4. **Compare** model performance

### 📊 **Export Results**
- Download allocation reports
- Export forecasting charts
- Save optimization parameters

## 🔧 Configuration

### **Environment Variables**
```bash
# Backend Configuration
PORT=8000                    # Server port
ENVIRONMENT=production       # Environment mode
CORS_ORIGINS=*              # CORS configuration

# Optional Redis
REDIS_URL=redis://localhost:6379
```

### **Railway Environment**
Railway automatically sets `PORT` - no additional configuration needed!

## 📖 API Documentation

### **Endpoints**
- `GET /health` - Health check
- `POST /api/optimize` - Run allocation optimization
- `GET /api/csv-data` - Fetch CSV data
- `POST /api/forecast` - ML forecasting

### **Interactive Docs**
Visit `/docs` on your deployed backend for interactive API documentation.

## 🧪 Testing

### **Frontend Tests**
```bash
cd frontend
npm test
```

### **Backend Tests**
```bash
cd backend/api/module4
pytest
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OptiU** - Branding and design inspiration
- **FastAPI** - Amazing Python web framework
- **React Team** - Excellent frontend framework
- **Railway** - Seamless deployment platform

## 📞 Support

- **📧 Issues**: [GitHub Issues](https://github.com/nourdesoukizz/allocation-maximizer/issues)
- **📚 Documentation**: [Project Wiki](https://github.com/nourdesoukizz/allocation-maximizer/wiki)
- **💬 Discussions**: [GitHub Discussions](https://github.com/nourdesoukizz/allocation-maximizer/discussions)

---

<div align="center">

**🚀 Ready to optimize your supply chain? Deploy now and start maximizing your allocation efficiency!**

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/nourdesoukizz/allocation-maximizer)

*Built with ❤️ using Claude Code*

</div>