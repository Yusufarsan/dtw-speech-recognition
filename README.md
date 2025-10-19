# DTW Speech Recognition System

A Python Speech Recognition System using Dynamic Time Warping (DTW) with Generalized Segmented Templates for vowel recognition.

## Description

This system implements a simple speech recognition system using DTW algorithm with generalized segmented templates to recognize Indonesian vowels (a, i, u, e, o). The system extracts 39-dimensional MFCC features (13 MFCC + Δ + ΔΔ) from audio samples and supports multiple audio formats (tested on .wav only). It uses generalized templates created by aligning training samples and segmenting them into k-segments for improved classification accuracy.

## Features

- **Multi-Format Audio Support**: WAV, MP3, FLAC, M4A, and OGG files
- **Advanced Template Creation**: Generalized segmented templates using k-segments approach
- **Feature Extraction**: 39D MFCC features using `python-speech-features`
  - 13 Mel-Frequency Cepstral Coefficients (MFCC)
  - Delta (Δ) coefficients (first derivative)
  - Delta-Delta (ΔΔ) coefficients (second derivative)
- **Multiple Distance Metrics**: 
  - Euclidean DTW distance
  - Mahalanobis distance (with covariance matrices)
  - Gaussian likelihood scoring
- **Evaluation Modes**:
  - Closed-set recognition (known vowels only)
  - Open-set recognition (with unknown rejection threshold)
  - Average accuracy reporting
- **Comprehensive Visualization Tools**:
  - Confusion matrices for performance analysis
  - DTW alignment path visualization
  - MFCC feature spectrograms
  - Distance distribution analysis
  - Template feature visualization
  - Threshold analysis for open-set recognition
  - Accuracy comparison charts

## Installation

### Requirements

- Python 3.6 or higher
- Dependencies listed in `requirements.txt`

### Install Dependencies

```bash
python -m venv my_venv
```

```bash
my_venv\Scripts\activate or source my_env_name/bin/activate
```

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy>=1.19.0`
- `scipy>=1.5.0`
- `python-speech-features>=0.6`
- `dtw-python>=1.1.0`
- `librosa>=0.8.0`
- `soundfile>=0.10.0`
- `matplotlib>=3.3.0`
- `seaborn>=0.11.0`
- `scikit-learn>=0.24.0`

## Dataset Structure

The system expects audio files organized in the following folder-based structure:

```
data/
├── train/
│   ├── a/
│   │   ├── sample1.wav
│   │   ├── sample2.wav
│   │   └── ...
│   ├── i/
│   │   └── ...
│   ├── u/
│   │   └── ...
│   ├── e/
│   │   └── ...
│   └── o/
│       └── ...
├── closed_test/
│   ├── a/
│   │   ├── test1.wav
│   │   └── ...
│   ├── i/
│   │   └── ...
│   ├── u/
│   │   └── ...
│   ├── e/
│   │   └── ...
│   └── o/
│       └── ...
└── open_test/
    ├── a/
    │   └── ...
    ├── i/
    │   └── ...
    ├── unknown/
    │   └── ...
    └── ...
```

- **train/**: Training data organized by vowel subdirectories. Used to create generalized templates.
- **closed_test/**: Closed-set test data organized by vowel subdirectories. Contains only known vowels.
- **open_test/**: Open-set test data organized by label subdirectories. Can contain both known vowels and unknown samples.
- **Audio formats**: Multiple formats supported: WAV, MP3, FLAC, M4A, OGG (recommended: 16 kHz, mono)

## Usage

### Basic Usage

Run the main program with visualization:

```bash
python main.py
```

Run the complete evaluation (closed-set, open-set, and average accuracy):

```bash
python dtw_recognizer.py
```

### Custom Parameters

Specify custom parameters for the DTW recognizer:

```bash
python dtw_recognizer.py --data-dir /path/to/your/data --threshold 500.0
```

### K-Segments Configuration

Initialize recognizer with custom k-segments:

```python
from dtw_recognizer import DTWRecognizer

# Use 3 segments per template for more detailed modeling
recognizer = DTWRecognizer(data_dir='data', k_segments=3)
```

### Available Distance Metrics

Choose a distance metric for classification:

```python
# Euclidean DTW (default)
recognizer.evaluate(distance_metric='euclidean')

# Mahalanobis distance with covariance
recognizer.evaluate(distance_metric='mahalanobis')

# Gaussian likelihood scoring
recognizer.evaluate(distance_metric='gaussian')
```

### Command Line Options

```bash
# Custom data directory
python dtw_recognizer.py --data-dir cupas-data

# Custom threshold for open-set evaluation
python dtw_recognizer.py --threshold 1500.0

# Combine options
python dtw_recognizer.py --data-dir cupas-data --threshold 800.0
```

## Visualization

The system includes comprehensive visualization tools for analyzing performance and understanding the DTW algorithm. **Note**: Visualizations are saved as PNG files in the `visualizations/` directory and are not displayed interactively.

### Quick Start

Run the complete visualization example:

```bash
python main.py
```

This generates all visualizations for multiple distance metrics and saves them in organized subdirectories:
- `visualizations/euclidean/`
- `visualizations/mahalanobis/`  
- `visualizations/gaussian/`

### Available Visualizations

1. **Confusion Matrix** - Shows classification accuracy per vowel
2. **DTW Alignment Path** - Visualizes how DTW warps sequences
3. **MFCC Features** - Displays audio waveform and spectrogram
4. **Distance Distribution** - Compares correct vs incorrect predictions
5. **Template Features** - Shows learned template patterns
6. **Accuracy Comparison** - Bar chart of performance metrics
7. **Threshold Analysis** - Distance distributions with threshold line

### Example Usage

```python
from dtw_recognizer import DTWRecognizer

# Initialize with custom parameters
recognizer = DTWRecognizer(data_dir='data', k_segments=2)
recognizer.load_templates()

# Evaluate with different metrics
results = recognizer.evaluate(distance_metric='euclidean')

# Generate visualizations (saved to files)
recognizer.visualize_confusion_matrix(
    results['closed_set'], 
    save_path='my_confusion_matrix.png'
)
recognizer.visualize_accuracy_comparison(
    results['closed_set'], 
    results['open_set'],
    save_path='accuracy_comparison.png'
)
```

### Documentation

For detailed visualization guide, see the comprehensive visualization outputs in the `visualizations/` folder after running `python main.py`.

### Example Output

```
============================================================
DTW SPEECH RECOGNITION SYSTEM - EVALUATION
============================================================
Loading training data and creating generalized segmented templates...
Supported formats: .wav, .mp3, .flac, .m4a, .ogg
Number of segments (k): 1
Loading vowel 'a': 5 training samples found
  Processing training samples for generalized template...
...

Total vowel templates created: 5

============================================================
CLOSED-SET EVALUATION
============================================================
Using distance metric: euclidean
✓ sample1.wav         | True: a | Predicted: a | Distance: 245.32
✓ sample2.wav         | True: a | Predicted: a | Distance: 198.76
...

Closed-set Accuracy: 23/25 = 92.00%

============================================================
OPEN-SET EVALUATION
============================================================
Using distance metric: euclidean
Unknown threshold: 1000.00

✓ sample1.wav         | True: a | Predicted: a | Distance: 245.32
✓ sample2.wav         | True: a | Predicted: a | Distance: 198.76
...

Open-set Accuracy: 22/25 = 88.00%

============================================================
SUMMARY
============================================================
K-Segments:          1
Distance Metric:     euclidean
Closed-set Accuracy: 92.00%
Open-set Accuracy:   88.00%
Average Accuracy:    90.00%
============================================================
```

## How It Works

1. **Template Creation**: Loads training data from `data/train/` directory and creates generalized segmented templates for each vowel
   - Aligns all training samples to a reference using DTW
   - Computes mean and covariance matrices for each vowel class
   - Segments the averaged template into k-segments for more detailed modeling
2. **Multi-Format Audio Support**: Extracts 39D MFCC features from various audio formats (WAV, MP3, FLAC, M4A, OGG)
3. **Advanced Classification**: Uses configurable distance metrics:
   - **Euclidean DTW**: Standard DTW distance against segmented templates
   - **Mahalanobis Distance**: Accounts for feature correlations using covariance matrices
   - **Gaussian Likelihood**: Probabilistic classification using multivariate Gaussian models
4. **Decision Making**: Selects the vowel with minimum distance/maximum likelihood
5. **Comprehensive Evaluation**: Reports accuracy for both closed-set (`data/closed_test/`) and open-set (`data/open_test/`) scenarios

## Algorithm Details

### Multi-Format Audio Processing

The system supports various audio formats through multiple audio loading backends:
- **Primary**: librosa (universal format support)
- **Fallback**: soundfile for high-quality audio I/O
- **Legacy**: scipy.wavfile for WAV files
- **Auto-normalization**: All audio is resampled to 16kHz mono for consistency

### MFCC Feature Extraction

The system extracts 39-dimensional features:
- 13 MFCCs using 26 filter banks and 512-point FFT
- 13 Delta (Δ) coefficients (first derivative)
- 13 Delta-Delta (ΔΔ) coefficients (second derivative)

### Generalized Segmented Templates

Advanced template creation process:
1. **Alignment**: All training samples are aligned to a reference using DTW
2. **Averaging**: Compute mean feature vectors and covariance matrices
3. **Segmentation**: Divide templates into k-segments for detailed modeling
4. **Storage**: Store both segmented means and covariance information

### Classification with Multiple Distance Metrics

The system supports multiple distance metrics:

1. **Euclidean DTW**: Standard DTW with Euclidean distance
   - Computes DTW distance between test sample and segmented template means
   - Fast and reliable for most applications

2. **Mahalanobis Distance**: Uses covariance information
   - Accounts for correlations between MFCC features
   - More robust to feature scaling variations
   - Requires sufficient training data for covariance estimation

3. **Gaussian Likelihood**: Probabilistic classification
   - Models each vowel class as multivariate Gaussian distribution
   - Uses mean vectors and covariance matrices from training data
   - Provides probability-based classification scores

### Open-Set Recognition

Enhanced open-set recognition with configurable thresholding:
- Calculates threshold for unknown rejection (default: 1000.0)
- Rejects samples with distance > threshold as "unknown"
- Supports samples in `open_test/unknown/` directory for evaluation
- Provides threshold analysis visualization tools

## Key Improvements

- **Multi-format audio support**: WAV, MP3, FLAC, M4A, OGG
- **Generalized segmented templates**: Improved accuracy through k-segment modeling
- **Multiple distance metrics**: Euclidean DTW, Mahalanobis, Gaussian likelihood
- **Advanced visualization**: Comprehensive analysis tools with file-based output
- **Robust audio loading**: Multiple fallback mechanisms for audio processing
- **Configurable parameters**: k-segments, distance metrics, thresholds

## Acknowledgments

- `python-speech-features` for MFCC extraction
- `dtw-python` for DTW implementation  
- `librosa` for universal audio format support
- `soundfile` for high-quality audio I/O
- `matplotlib` and `seaborn` for visualization
- `scikit-learn` for evaluation metrics
- Dynamic Time Warping algorithm for speech recognition
