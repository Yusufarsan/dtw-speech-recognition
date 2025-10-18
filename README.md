# DTW Speech Recognition System

A Python 3 Speech Recognition System using Dynamic Time Warping (DTW) for vowel recognition.

## Description

This system implements a simple speech recognition system using DTW algorithm to recognize Indonesian vowels (a, i, u, e, o). The system extracts 39-dimensional MFCC features (13 MFCC + Δ + ΔΔ) from audio samples and uses DTW for classification.

## Features

- **Feature Extraction**: 39D MFCC features using `python-speech-features`
  - 13 Mel-Frequency Cepstral Coefficients (MFCC)
  - Delta (Δ) coefficients (first derivative)
  - Delta-Delta (ΔΔ) coefficients (second derivative)
- **Classification**: DTW-based matching using `dtw-python`
- **Evaluation Modes**:
  - Closed-set recognition (known vowels only)
  - Open-set recognition (with unknown rejection threshold)
  - Average accuracy reporting
- **Visualization**: Comprehensive visualization tools
  - Confusion matrices for performance analysis
  - DTW alignment path visualization
  - MFCC feature spectrograms
  - Distance distribution analysis
  - Template feature visualization
  - Threshold analysis for open-set recognition

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
- **Audio format**: WAV files (recommended: 16 kHz, mono)

## Usage

### Basic Usage

Run the complete evaluation (closed-set, open-set, and average accuracy):

```bash
python dtw_recognizer.py
```

### Custom Data Directory

Specify a custom data directory:

```bash
python dtw_recognizer.py --data-dir /path/to/your/data
```

### Custom Threshold for Open-Set

Set a custom distance threshold for open-set evaluation:

```bash
python dtw_recognizer.py --threshold 500.0
```

### Custom Distance Metric

Choose a distance metric for classification:

```bash
python dtw_recognizer.py --distance-metric euclidean
python dtw_recognizer.py --distance-metric mahalanobis
python dtw_recognizer.py --distance-metric gaussian
python dtw_recognizer.py --distance-metric negative_gaussian
```

Note: Mahalanobis and Gaussian-based metrics require implementation of mean and covariance calculation.

## Visualization

The system includes comprehensive visualization tools for analyzing performance and understanding the DTW algorithm.

### Quick Start

Run the complete visualization example:

```bash
python visualization_example.py
```

This generates all visualizations and saves them in the `visualizations/` directory.

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

# Initialize and evaluate
recognizer = DTWRecognizer(data_dir='data')
recognizer.load_templates()
results = recognizer.evaluate()

# Generate visualizations
recognizer.visualize_confusion_matrix(results['closed_set'])
recognizer.visualize_accuracy_comparison(results['closed_set'], results['open_set'])
recognizer.visualize_distance_distribution(results['closed_set'])
```

### Documentation

For detailed visualization guide, see [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)

### Example Output

```
============================================================
DTW SPEECH RECOGNITION SYSTEM - EVALUATION
============================================================
Loading templates...
  Loaded template_p1.wav for vowel 'a'
  Loaded template_p2.wav for vowel 'a'
  ...

Total templates loaded: 25

============================================================
CLOSED-SET EVALUATION
============================================================
✓ uji_p1.wav          | True: a | Predicted: a | Distance: 245.32
✓ uji_p2.wav          | True: a | Predicted: a | Distance: 198.76
...

Closed-set Accuracy: 23/25 = 92.00%

============================================================
OPEN-SET EVALUATION
============================================================
Unknown threshold: 450.00

✓ uji_p1.wav          | True: a | Predicted: a | Distance: 245.32
✓ uji_p2.wav          | True: a | Predicted: a | Distance: 198.76
...

Open-set Accuracy: 22/25 = 88.00%

============================================================
SUMMARY
============================================================
Closed-set Accuracy: 92.00%
Open-set Accuracy:   88.00%
Average Accuracy:    90.00%
============================================================
```

## How It Works

1. **Template Loading**: Loads training data from `data/train/` directory and creates generalized templates for each vowel using mean and covariance
2. **Feature Extraction**: Extracts 39D MFCC features from each audio file
3. **Classification**: Uses configurable distance metrics (DTW, Mahalanobis, Gaussian) to compare test features with templates
4. **Decision**: Selects the vowel with minimum distance/maximum likelihood
5. **Evaluation**: Reports accuracy for both closed-set (from `data/closed_test/`) and open-set (from `data/open_test/`) scenarios

## Algorithm Details

### MFCC Feature Extraction

The system extracts 39-dimensional features:
- 13 MFCCs using 26 filter banks and 512-point FFT
- 13 Delta (Δ) coefficients
- 13 Delta-Delta (ΔΔ) coefficients

### Classification with Multiple Distance Metrics

The system supports multiple distance metrics:

1. **Euclidean (DTW)**: Standard DTW with Euclidean distance
   - Computes DTW distance between test sample and all training samples
   - Selects the vowel class with minimum DTW distance

2. **Mahalanobis Distance**: Uses mean template and covariance matrix
   - Accounts for correlations between features
   - Requires implementation of mean and covariance calculation

3. **Gaussian Likelihood**: Probabilistic approach using Gaussian distribution
   - Uses mean and covariance to model each vowel class
   - Requires implementation of mean and covariance calculation

4. **Negative Gaussian Log-Likelihood**: Distance-based version of Gaussian likelihood
   - Lower values indicate better match
   - Requires implementation of mean and covariance calculation

### Open-Set Recognition

- Calculates a threshold based on median intra-class distances
- Rejects samples with distance > threshold as "unknown"
- Default threshold: 1.5 × median of intra-class template distances
- Test files in `open_test/unknown/` directory are expected to be rejected

## Acknowledgments

- `python-speech-features` for MFCC extraction
- `dtw-python` for DTW implementation
- Dynamic Time Warping algorithm for speech recognition
