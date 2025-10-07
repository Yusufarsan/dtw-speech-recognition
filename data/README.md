# Data Directory Structure

This directory contains audio samples for vowel recognition.

## Structure
- Each vowel (a, i, u, e, o) has its own subdirectory
- Template files: `template_p1.wav`, `template_p2.wav`, ..., `template_p5.wav`
- Test files: `uji_p1.wav`, `uji_p2.wav`, ..., `uji_pN.wav`

## File Format
- Audio format: WAV
- Sampling rate: 16 kHz (recommended)
- Channels: Mono

## Example
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
└── ...
```
