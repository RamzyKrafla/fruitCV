# 🍎 CompVis Fruit Classification

A complete computer vision project for fruit classification using deep learning. This project includes dataset preparation, model training, fine-tuning, and a beautiful React frontend for in-browser inference.

## 🌟 Features

- **Multi-class Fruit Classification** - Classifies 5 different fruit types
- **Transfer Learning** - Uses pre-trained ResNet18 for better accuracy
- **Fine-tuning** - Advanced training with learning rate scheduling
- **In-browser Inference** - React frontend with ONNX runtime
- **PDF Support** - Upload and classify PDF documents
- **Beautiful UI** - Modern, responsive design with drag-and-drop
- **No Backend Required** - Pure frontend solution using WebAssembly

## 🍊 Supported Fruit Classes

- 🍎 Apple
- 🍌 Banana
- 🍊 Orange
- 🍓 Strawberry
- 🥭 Mango

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- Kaggle API credentials (for dataset download)

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd CompVis
```

### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Kaggle API Setup

```bash
# Create Kaggle credentials directory
mkdir ~/.kaggle

# Download kaggle.json from https://www.kaggle.com/account
# Place it in ~/.kaggle/kaggle.json

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Dataset Preparation

```bash
# Download and prepare fruit dataset
python download_fruits.py

# Clean corrupted images (if needed)
python clean_dataset.py
```

### 5. Model Training

```bash
# Train initial model
python train.py

# Fine-tune for better accuracy
python finetune.py

# Export to ONNX for web deployment
python export_onnx.py
```

### 6. Frontend Setup

```bash
cd ui
npm install
npm run dev
```

Open your browser to `http://localhost:5174` and start classifying fruits!

## 📁 Project Structure

```
CompVis/
├── data/                   # Dataset directory
│   ├── raw/               # Downloaded images
│   └── fruit_subset/      # Processed train/val/test splits
├── model/                 # Trained models
│   ├── best_resnet18_finetune.onnx
│   └── classes.json
├── ui/                    # React frontend
│   ├── src/
│   │   ├── App.jsx        # Main application
│   │   ├── App.css        # Styles
│   │   └── infer.js       # ONNX inference logic
│   └── public/
│       └── model/         # ONNX model files
├── test_images/           # Sample images for testing
├── download_fruits.py     # Dataset download script
├── train.py              # Initial model training
├── finetune.py           # Model fine-tuning
├── export_onnx.py        # ONNX export
├── predict.py            # Command-line prediction
├── clean_dataset.py      # Dataset cleaning utility
└── requirements.txt      # Python dependencies
```

## 🧠 Model Architecture

- **Backbone**: ResNet18 (pre-trained on ImageNet)
- **Classifier**: Custom fully connected layer
- **Input**: 224x224 RGB images
- **Output**: 5-class softmax probabilities
- **Training**: Transfer learning with fine-tuning

## 🎯 Training Process

### Phase 1: Initial Training

- Freeze ResNet18 backbone
- Train only the classifier head
- 15 epochs with Adam optimizer
- Learning rate: 0.001

### Phase 2: Fine-tuning

- Unfreeze last ResNet18 block
- Train with reduced learning rate
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting

## 🌐 Frontend Features

- **Drag & Drop Upload** - Easy file selection
- **Multiple File Types** - JPEG, JPG, PNG, PDF
- **Real-time Inference** - Instant classification results
- **Visual Results** - Top prediction with emoji and confidence
- **Probability Bars** - All class probabilities displayed
- **Responsive Design** - Works on desktop and mobile
- **No Backend** - Pure client-side processing

## 🔧 Technical Details

### Backend (Python)

- **Framework**: PyTorch
- **Model**: ResNet18 with custom classifier
- **Data Augmentation**: Random crop, flip, color jitter
- **Optimization**: Adam with learning rate scheduling
- **Export**: ONNX format for web deployment

### Frontend (React)

- **Framework**: React 18 with Vite
- **Inference**: ONNX Runtime Web (WebAssembly)
- **PDF Processing**: PDF.js for document rendering
- **Styling**: Modern CSS with glass morphism effects
- **Build Tool**: Vite for fast development

## 📊 Performance

- **Training Accuracy**: ~95% on validation set
- **Inference Speed**: <100ms per image (browser)
- **Model Size**: ~45MB (ONNX format)
- **Memory Usage**: ~50MB (browser)

## 🛠️ Usage Examples

### Command Line Prediction

```bash
python predict.py --image path/to/fruit.jpg
```

### Web Interface

1. Open `http://localhost:5174`
2. Drag and drop fruit images or PDFs
3. Click "Predict" to see results
4. View top prediction and all probabilities

## 🔒 Security

- API keys stored in `~/.kaggle/` (not in project)
- `.gitignore` protects sensitive files
- No backend server required
- All processing happens locally

## 🐛 Troubleshooting

### Common Issues

**SSL Certificate Error**

```bash
# Install certificates
pip install certifi
# Run certificate installer
/Applications/Python\ 3.x/Install\ Certificates.command
```

**Multiprocessing Error (macOS)**

- Set `NUM_WORKERS = 0` in training scripts
- Use `if __name__ == '__main__':` guards

**WASM Loading Error**

- Ensure Vite config includes WASM support
- Check browser console for detailed errors
- Try refreshing with hard reload (Ctrl+F5)

**Model Loading Error**

- Verify ONNX model exists in `ui/public/model/`
- Check `classes.json` file format
- Re-export model if needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Fruit images from various sources
- **Model**: ResNet18 from torchvision
- **Frontend**: React and ONNX Runtime Web
- **Icons**: Fruit emojis for visual appeal

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section
2. Review browser console for errors
3. Verify all dependencies are installed
4. Ensure proper file permissions

---

**Happy Fruit Classifying! 🍎🍌🍊🍓🥭**
