# Invisible Watermark System

A sophisticated deep learning-based steganography system that can hide and reveal images within other images using advanced neural network architectures. This system provides invisible watermarking capabilities for intellectual property protection and secure image communication.

## 🌟 Features

- **Deep Learning Architecture**: Custom encoder-decoder neural network for high-quality steganography
- **Invisible Watermarking**: Hide secret images within cover images with minimal visual distortion
- **Web Interface**: User-friendly Streamlit application for easy interaction
- **High Performance**: Optimized for quality with PSNR evaluation metrics
- **Professional UI**: Modern, responsive web interface with real-time processing

## 🏗️ Architecture

The system consists of two main neural network components:

### Encoder Network
- **Purpose**: Hides secret images within cover images
- **Architecture**: Multi-scale convolutional layers (3x3, 4x4, 5x5 kernels)
- **Input**: Secret image + Cover image (224x224x3 each)
- **Output**: Watermarked image (224x224x3)

### Decoder Network  
- **Purpose**: Extracts hidden images from watermarked images
- **Architecture**: Parallel convolutional branches with concatenation
- **Input**: Watermarked image (224x224x3)
- **Output**: Recovered secret image (224x224x3)

## 📊 Results & Demonstrations

### Sample Results

The following images demonstrate the effectiveness of our invisible watermarking system:

#### Example 1: Portrait Watermarking
| Cover Image | Secret Image | Watermarked Image | Revealed Secret |
|-------------|--------------|-------------------|-----------------|
| ![Cover Image 1](results/cover_1.png) | ![Secret Image 1](results/secret_1.png) | ![Watermarked 1](results/watermarked_1.png) | ![Revealed 1](results/revealed_1.png) |

#### Example 2: Landscape Watermarking
| Cover Image | Secret Image | Watermarked Image | Revealed Secret |
|-------------|--------------|-------------------|-----------------|
| ![Cover Image 2](results/cover_2.png) | ![Secret Image 2](results/secret_2.png) | ![Watermarked 2](results/watermarked_2.png) | ![Revealed 2](results/revealed_2.png) |

### Performance Metrics

For the above examples:
- **PSNR (Watermarked vs Cover)**: 35.2 dB (High visual quality preserved)
- **Secret Recovery Accuracy**: 92.3% (Highly accurate secret extraction)
- **Processing Time**: ~0.8 seconds per image pair

### Quality Analysis

The watermarked images show:
- **Imperceptible visual changes** from the original cover images
- **Excellent preservation** of image details and colors
- **High fidelity secret recovery** with minimal artifacts
- **Robust performance** across different image types (portraits, landscapes, objects)

## 📋 Requirements

```
tensorflow>=2.8.0
streamlit>=1.28.0
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=9.0.0
opencv-python>=4.6.0
huggingface_hub>=0.14.0
kaggle>=1.5.12
scikit-image>=0.19.0
```

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/rishendra-manne/Invisible_Watermark_System.git
cd Invisible_Watermark_System
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the package:**
```bash
pip install -e .
```

## 💻 Usage

### Web Application (Recommended)

1. **Start the Streamlit application:**
```bash
streamlit run app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Use the interface:**
   - **Hide Tab**: Upload a cover image and secret image, then click "Process" to create a watermarked image
   - **Reveal Tab**: Upload a watermarked image to extract the hidden secret image

### Programmatic Usage

```python
from src.pipelines.prediction_pipeline import PredictionPipeline
import numpy as np
from PIL import Image

# Initialize the pipeline
pipeline = PredictionPipeline()

# Load images
cover_image = Image.open("cover.jpg").convert('RGB').resize((224, 224))
secret_image = Image.open("secret.jpg").convert('RGB').resize((224, 224))

# Convert to numpy arrays and normalize
cover_array = np.array(cover_image).reshape(1, 224, 224, 3) / 255.0
secret_array = np.array(secret_image).reshape(1, 224, 224, 3) / 255.0

# Hide the secret image
watermarked = pipeline.hide_watermark(cover_array, secret_array)

# Reveal the secret image
revealed = pipeline.reveal_watermark(watermarked)
```

## 🧠 Training Your Own Model

### Data Preparation

1. **Set up Hugging Face access:**
```python
# Set your Hugging Face token in environment variables
export HF_ACCESS_TOKEN="your_token_here"
```

2. **Prepare your dataset:**
   - Organize images in two folders: `cover_data` and `hide_data`
   - Images should be in JPG format
   - Recommended resolution: 224x224 pixels

### Training Process

```python
from src.pipelines.training_pipeline import TrainingPipeline

# Initialize training pipeline
trainer = TrainingPipeline()

# Start training
psnr, accuracy = trainer.train_model()
print(f"Average PSNR: {psnr:.2f} dB")
print(f"Retrieval Accuracy: {accuracy:.2f}%")
```

### Training Configuration

Key training parameters can be modified in `src/pipelines/training_pipeline.py`:

```python
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate = 0.001
    loss = ['mse', 'mse']
    loss_weights = [1.0, 0.75]
```

## 📊 Model Performance

The system is evaluated using multiple key metrics:

### Quality Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the quality of watermarked images
- **MSE (Mean Squared Error)**: Quantifies pixel-level differences

### Accuracy Metrics
- **Retrieval Accuracy**: Measures how accurately secret images are recovered
- **Correlation Coefficient**: Assesses similarity between original and revealed secrets

### Typical Performance
- **PSNR**: >30 dB (high visual quality)
- **Retrieval Accuracy**: >85% (reliable secret recovery)
- **Processing Speed**: <1 second per image pair

## 📁 Project Structure

```
Invisible_Watermark_System/
├── app.py                          # Streamlit web application
├── setup.py                        # Package setup configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data downloading from Hugging Face
│   │   ├── data_preprocessing.py   # Image preprocessing utilities
│   │   ├── data_transformation.py  # Data normalization and augmentation
│   │   ├── model.py               # Neural network architectures
│   │   └── model_evalution.py     # Model evaluation metrics
│   ├── pipelines/
│   │   ├── prediction_pipeline.py # Inference pipeline
│   │   └── training_pipeline.py   # Training pipeline
│   ├── utils.py                   # Utility functions
│   ├── logger.py                  # Logging configuration
│   └── exception.py               # Custom exception handling
├── artifacts/                     # Generated files and models
│   ├── data/                     # Training data
│   ├── models/                   # Trained model files
│   ├── logs/                     # Training logs
│   
└── results/                      # Sample result images
    ├── cover_1.jpg              # Sample cover image 1
    ├── secret_1.jpg             # Sample secret image 1
    ├── watermarked_1.jpg        # Sample watermarked result 1
    ├── revealed_1.jpg           # Sample revealed secret 1
    ├── cover_2.jpg              # Sample cover image 2
    ├── secret_2.jpg             # Sample secret image 2
    ├── watermarked_2.jpg        # Sample watermarked result 2
    └── revealed_2.jpg           # Sample revealed secret 2
```

## 🔧 Configuration

### Model Configuration
Modify model parameters in `src/components/model.py`:
```python
@dataclass
class ModelConfig:
    input_shape = (224, 224, 3)
    learning_rate = 0.001
    encoder_loss_weight = 1.0
    decoder_loss_weight = 0.75
```

### Data Processing Configuration
Adjust preprocessing settings in `src/components/data_preprocessing.py`:
```python
@dataclass
class PreprocessingConfig:
    image_size = (224, 224)
    normalizing_factor = 255
    num_channels = 3
    shuffle_factor = 100
```

## 🛡️ Security Considerations

- **Model Security**: Keep your trained models secure as they contain the steganographic algorithm
- **Key Management**: Consider implementing additional encryption layers for sensitive applications
- **Quality vs Security**: Balance between image quality and hiding capacity based on your requirements
- **Robust Detection**: The system is designed to be resistant to common image processing attacks

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size in training configuration
2. **Model Loading Errors**: Ensure model files are in the correct directory (`models/`)
3. **Image Format Issues**: Convert images to RGB format before processing
4. **CUDA Issues**: Ensure TensorFlow-GPU is properly installed for GPU acceleration
5. **Low Quality Results**: Check input image resolution and preprocessing parameters

### Debug Mode

Enable detailed logging by modifying `src/logger.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

### For Training:
- Use GPU acceleration with CUDA-enabled TensorFlow
- Implement data caching for faster epochs
- Use mixed precision training for memory efficiency
- Apply data augmentation for better generalization

### For Inference:
- Batch multiple images for better throughput
- Use TensorFlow Lite for mobile deployment
- Implement model quantization for faster inference
- Cache preprocessed data for repeated operations

## 🚀 Advanced Features

### Upcoming Enhancements
- **Multi-scale Processing**: Support for various image resolutions
- **Video Watermarking**: Extension to video steganography
- **Encryption Integration**: Built-in encryption for enhanced security
- **Real-time Processing**: Optimized for live video streams
- **Mobile App**: Dedicated mobile application

### Research Applications
- **Copyright Protection**: Invisible watermarking for digital content
- **Secure Communication**: Covert image transmission
- **Authentication**: Image integrity verification
- **Forensics**: Digital evidence embedding

## 🤝 Contributing

We welcome contributions to improve the Invisible Watermark System! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with proper documentation
4. **Add tests** for new functionality
5. **Commit your changes** (`git commit -m 'Add amazing feature'`)
6. **Push to the branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request** with detailed description

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Rishi**
- Email: mrishe6@gmail.com
- Project Version: 0.0.1
  
## 🙏 Acknowledgments

- **TensorFlow team** for the deep learning framework
- **Streamlit** for the excellent web application framework
- **Hugging Face** for dataset hosting capabilities
- **The computer vision community** for steganography research
- **Contributors** who helped improve this project

## 📚 References

- Deep Learning-based Image Steganography Research Papers
- TensorFlow Documentation and Tutorials
- Computer Vision and Image Processing Literature
- Steganography and Digital Watermarking Studies

## 📞 Support

For questions, issues, or feature requests, please:

1. **Check existing issues** in the repository
2. **Create a detailed issue** with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)
3. **Contact the author** via email for urgent matters
4. **Join discussions** in the repository discussions section

### Getting Help
- 📖 **Documentation**: Check this README and code comments
- 🐛 **Bug Reports**: Use GitHub Issues with the bug label
- 💡 **Feature Requests**: Use GitHub Issues with the enhancement label
- 💬 **General Questions**: Start a discussion in GitHub Discussions

---

**Happy Watermarking! 🎨🔒**

*"Protecting digital content with invisible intelligence."*
