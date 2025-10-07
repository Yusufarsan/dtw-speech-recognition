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

## Dataset Structure

The system expects audio files organized in the following structure:

```
data/
├── a/
│   ├── template_p1.wav
│   ├── template_p2.wav
│   ├── template_p3.wav
│   ├── template_p4.wav
│   ├── template_p5.wav
│   ├── uji_p1.wav
│   └── uji_p2.wav
├── i/
│   ├── template_p1.wav
│   └── ...
├── u/
│   ├── template_p1.wav
│   └── ...
├── e/
│   ├── template_p1.wav
│   └── ...
└── o/
    ├── template_p1.wav
    └── ...
```

- **Template files**: `template_p1.wav` through `template_p5.wav` (P1-P5)
- **Test files**: `uji_pY.wav` (where Y is any number)
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

1. **Template Loading**: Loads 5 template files (P1-P5) for each vowel
2. **Feature Extraction**: Extracts 39D MFCC features from each audio file
3. **Classification**: Uses DTW to compare test features with all templates
4. **Decision**: Selects the vowel with minimum DTW distance
5. **Evaluation**: Reports accuracy for both closed-set and open-set scenarios

## Algorithm Details

### MFCC Feature Extraction

The system extracts 39-dimensional features:
- 13 MFCCs using 26 filter banks and 512-point FFT
- 13 Delta (Δ) coefficients
- 13 Delta-Delta (ΔΔ) coefficients

### DTW Classification

- Computes DTW distance between test sample and all templates
- Uses Euclidean distance as the local distance metric
- Selects the vowel class with minimum DTW distance

### Open-Set Recognition

- Calculates a threshold based on median template distances
- Rejects samples with distance > threshold as "unknown"
- Default threshold: 1.5 × median of intra-class template distances

## Acknowledgments

- `python-speech-features` for MFCC extraction
- `dtw-python` for DTW implementation
- Dynamic Time Warping algorithm for speech recognition
