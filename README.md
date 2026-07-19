# CHTN (Capybla Harmonic Transmission)

> **Experimental parametric audio format based on real-time additive synthesis.**

![Status](https://img.shields.io/badge/status-experimental-blue)
![Language](https://img.shields.io/badge/python-3.12+-green)
![License](https://img.shields.io/badge/license-Non--Commercial-red)

---

## Overview

CHTN is an experimental audio format that stores a **parametric description** of sound instead of raw audio samples.

Rather than encoding waveforms directly like MP3 or AAC, CHTN analyzes the audio and stores the information required for a synthesis engine to reconstruct it in real time.

The project includes:

- CHTN Studio
- Encoder
- Decoder
- Real-time DSP engine
- Dynamic oscillator allocation
- Audio visualizers
- Experimental container format

---

## How it works

Traditional audio formats store compressed audio samples.

CHTN follows a different philosophy.

```
Original Audio
       │
       ▼
Spectral Analysis
       │
       ▼
Oscillator + Noise Parameters
       │
       ▼
CHTN Container
       │
       ▼
Real-Time DSP Engine
       │
       ▼
Reconstructed Audio
```

Instead of storing every audio sample, CHTN stores the information needed to recreate the sound.

---

## Features

- Real-time additive synthesis
- Dynamic oscillator allocation
- Parametric audio representation
- Multi-threaded playback engine
- Noise synthesis support
- Stereo playback
- Audio visualizers
- Modern CHTN container
- Backward compatibility between format revisions
- Experimental adaptive compression

---

## Current Status

Current version:

```
CHTN Studio v31
```

Current capabilities:

- Stable real-time playback
- Dynamic oscillator engine
- Encoder and decoder
- CHTN Studio GUI
- Compatible with:

```
MP3
WAV
FLAC
OGG
M4A
```

---

## Project Structure

```
CHTN/
│
├── engine/
│   ├── encoder
│   ├── decoder
│   └── DSP engine
│
├── studio/
│   └── CHTN Studio
│
├── examples/
│   ├── example1.chtn
│   ├── example2.chtn
│   └── example3.chtn
│
├── docs/
│   └── format documentation
│
├── images/
│
├── README.md
└── LICENSE
```

---

## Compression

CHTN does **not** use a fixed bitrate.

File size depends on the spectral complexity of the source audio.

Typical behavior:

| Audio Type | Compression |
|------------|------------|
| Voice | Excellent |
| Chiptune | Excellent |
| Piano | Very Good |
| Orchestra | Good |
| Rock | Moderate |
| White Noise | Poor |

---

## Performance

Designed for real-time decoding.

Typical CPU usage depends on:

- Number of active oscillators
- Audio complexity
- Block size
- Available CPU resources

Playback quality scales with the number of oscillators.

---

## Requirements

Python 3.12+

Libraries used by CHTN Studio:

- numpy
- scipy
- sounddevice
- tkinter
- librosa
- matplotlib

(Some versions may require additional dependencies.)

---

## Example

```
Input:
music.mp3

↓

CHTN Encoder

↓

music.chtn

↓

Real-Time Decoder

↓

Audio Output
```

---

## Philosophy

CHTN is an experiment exploring a different approach to digital audio.

Instead of asking:

> "How can we compress samples better?"

it asks:

> "How can we describe the sound itself?"

The goal is not to replace existing codecs, but to explore the possibilities of real-time parametric audio reconstruction.

---

## Roadmap

Planned improvements include:

- Better low-frequency reconstruction
- Improved container efficiency
- Enhanced visualizers
- More DSP optimizations
- Better documentation
- Native CHTN specification
- Cross-platform support

---

## License

This project is free to use for personal and educational purposes.

You may:

- Use
- Study
- Modify
- Share

You must:

- Give appropriate credit to the original author.

Commercial use is **not permitted** without explicit permission from the author.

See the LICENSE file for details.

---

## Author

**David Hernández (Capybla)**

GitHub:

https://github.com/Capybla

---

## Acknowledgements

CHTN was developed as an independent experimental project exploring additive synthesis, digital signal processing, and alternative approaches to audio representation.

---

*CHTN is an experimental project and is under active development.*
