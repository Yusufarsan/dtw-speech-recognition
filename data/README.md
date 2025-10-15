# Data Directory Structure

This directory contains audio samples for vowel recognition organized in a folder-based structure.

## Structure

The data directory should contain three main subdirectories:

### 1. train/
Training data for creating generalized templates. Each vowel has its own subdirectory containing training audio files.

```
train/
├── a/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
├── i/
├── u/
├── e/
└── o/
```

### 2. closed_test/
Closed-set test data containing only known vowels. Each vowel has its own subdirectory.

```
closed_test/
├── a/
│   ├── test1.wav
│   └── ...
├── i/
├── u/
├── e/
└── o/
```

### 3. open_test/
Open-set test data that can contain both known vowels and unknown samples. Each label has its own subdirectory.

```
open_test/
├── a/
│   └── ...
├── i/
│   └── ...
├── unknown/
│   └── ...
└── ...
```

## File Format
- Audio format: WAV (or MP3, FLAC, M4A, OGG)
- Sampling rate: 16 kHz (recommended)
- Channels: Mono

## Notes
- File names can be arbitrary - the system reads all audio files from each directory
- The directory name determines the label (e.g., files in `train/a/` are labeled as vowel 'a')
- The `unknown` directory in `open_test/` should contain samples that don't belong to any known vowel class
