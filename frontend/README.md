# VeriFace AI - Deepfake Detector

A full-stack application that detects whether facial images are real or AI-generated using deep learning. The application features a FastAPI backend with a trained ResNet model and an interactive React frontend.

## 🎯 Features

- **AI-Generated Face Detection**: Uses a deep learning model to identify synthetic/AI-generated faces
- **Interactive Web Interface**: Drag-and-drop image upload with real-time analysis
- **REST API**: FastAPI backend with automatic documentation
- **CUDA Support**: GPU acceleration when available
- **CORS Enabled**: Ready for seamless frontend-backend integration

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- 2GB+ disk space (for model weights)
- GPU (optional, but recommended for faster inference)

## 🚀 Quick Start

### Backend Setup

1. Navigate to the project root directory
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   python src/app.py
   ```
   The API will be available at `http://localhost:8000`
   API documentation: `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The app will be available at `http://localhost:5173`

## 📁 Project Structure

```
deepfake-detector/
├── src/                    # Backend (FastAPI)
│   ├── app.py             # Main API application
│   ├── train.py           # Model training script
│   ├── check_csv.py       # Data validation utilities
│   └── download_model.py  # Model download utility
├── models/                # Trained model weights
│   └── best_model.pth     # ResNet-based detector model
├── data/                  # Training datasets
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   └── real_vs_fake/      # Image dataset
├── frontend/              # React + Vite UI
│   ├── src/
│   ├── public/
│   └── vite.config.js
└── requirements.txt       # Python dependencies
```

## 🔧 API Endpoints

### POST `/predict`
Analyze an image for deepfake detection.

**Request:**
```json
{
  "image": "base64_encoded_image_data"
}
```

**Response:**
```json
{
  "prediction": "real" | "fake",
  "confidence": 0.95
}
```

## 🎨 Frontend Usage

1. Open the web application
2. Drag and drop an image or click to upload
3. Wait for analysis to complete
4. View the detection result with confidence score

## 📊 Available Scripts

### Backend
- `python src/train.py` - Train the model on dataset
- `python src/check_csv.py` - Validate dataset CSV files
- `python src/download_model.py` - Download pre-trained model

### Frontend
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint checks

## 🛠️ Technology Stack

**Backend:**
- FastAPI
- PyTorch
- TorchVision (ResNet)
- Pillow (Image processing)
- OpenCV

**Frontend:**
- React 19
- Vite
- Axios
- Framer Motion
- Lucide React
- React Dropzone

## 📦 Model Information

The detection model is based on ResNet architecture, trained to classify images as:
- **Real**: Authentic photographs
- **Fake**: AI-generated/synthetic images

Model is automatically downloaded on first run from Google Drive if not present locally.

## 🐛 Troubleshooting

**Model download fails:**
- Check internet connection
- Ensure `models/` directory has write permissions
- Try manual download from Google Drive

**CORS errors:**
- Ensure backend is running on `http://localhost:8000`
- Check that CORS middleware is enabled in `app.py`

**GPU not detected:**
- Install CUDA toolkit for your GPU
- Verify PyTorch CUDA version matches your setup

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!
