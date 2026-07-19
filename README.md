# CHTN (Capybla Harmonic Transmission)

> **Experimental parametric audio format based on real-time additive synthesis.**

🌍 **Languages**

- 🇬🇧 **English** (this file)
- 🇪🇸 [Español](README.es.md)

![Status](https://img.shields.io/badge/status-experimental-blue)
![Python](https://img.shields.io/badge/python-3.12+-green)
![License](https://img.shields.io/badge/license-NC--Attribution-red)

---

## Overview

CHTN (**Capybla Harmonic Transmission**) is an experimental audio format based on **real-time additive synthesis**.

Unlike traditional codecs such as MP3, AAC or FLAC, CHTN does not primarily store compressed audio samples. Instead, it stores a compact **parametric description** of the sound that is reconstructed during playback by a lightweight DSP engine.

The project includes:

- 🎵 CHTN Studio
- ⚙️ Encoder
- 🔊 Decoder
- 🧠 Real-time DSP engine
- 🎚 Dynamic oscillator allocation
- 📊 Audio visualizers
- 📦 CHTN container format

---

## How it Works

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

Instead of storing every individual audio sample, CHTN stores the information required to synthesize the sound in real time.

---

## Features

- Real-time additive synthesis
- Dynamic oscillator allocation
- Parametric audio representation
- Multi-threaded playback
- Noise synthesis
- Stereo support
- Audio visualizers
- Adaptive compression
- Backward-compatible container revisions
- Experimental DSP architecture

---

## Project Status

Current version:

```
CHTN Studio v31
```

Current capabilities:

- Real-time playback
- Audio encoding
- Audio decoding
- CHTN Studio GUI
- Dynamic oscillator engine
- Compatible with:

- MP3
- WAV
- FLAC
- OGG
- M4A

---

## Philosophy

Most audio formats ask:

> *"How can audio samples be compressed more efficiently?"*

CHTN asks a different question:

> *"How can the sound itself be described mathematically and reconstructed later?"*

The objective of this project is **not to replace existing codecs**, but to explore an alternative approach to digital audio representation through parametric synthesis.

---

## Compression

CHTN does **not** use a fixed bitrate.

File size depends on the spectral complexity of the source material.

Typical behavior:

| Audio | Typical Compression |
|--------|---------------------|
| Speech | Excellent |
| Chiptune | Excellent |
| Piano | Very Good |
| Orchestra | Good |
| Rock / Metal | Moderate |
| White Noise | Poor |

---

## License

This project is released under a **non-commercial attribution license**.

You may:

- Use
- Study
- Modify
- Share

You must:

- Credit the original author.

Commercial use requires explicit permission from the author.

---

## Author

**David Hernández (Capybla)**

GitHub:

https://github.com/Capybla

---

*CHTN is an independent experimental research project focused on additive synthesis, DSP and alternative audio representation.*
