# LatentSync-rs

[![CI](https://github.com/bhubbard/latentsync-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/bhubbard/latentsync-rs/actions/workflows/ci.yml)
[![Code Coverage](https://github.com/bhubbard/latentsync-rs/actions/workflows/coverage.yml/badge.svg)](https://github.com/bhubbard/latentsync-rs/actions/workflows/coverage.yml)
[![Deploy GitHub Pages](https://github.com/bhubbard/latentsync-rs/actions/workflows/pages.yml/badge.svg)](https://code.brandonhubbard.com/latentsync-rs/)
[![arXiv](https://img.shields.io/badge/arXiv-2412.09262-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.09262)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021_edition-orange.svg)](Cargo.toml)

> **High-performance, memory-safe pure Rust port of ByteDance's LatentSync 1.6: audio-conditioned latent diffusion for temporally consistent lip-sync video synthesis with SyncNet supervision.**

🌐 **Live Interactive Latent Diffusion & Cross-Attention Studio**: **[https://code.brandonhubbard.com/latentsync-rs/](https://code.brandonhubbard.com/latentsync-rs/)**  
📊 **Full Empirical Performance Benchmarks**: **[BENCHMARKS.md](BENCHMARKS.md)** (**40.0x faster** Whisper feature extraction & SafeTensors inspection, **36.1x faster** end-to-end pipeline vs. Python)

---

## 📖 Overview

**LatentSync-rs** implements an end-to-end lip-sync pipeline based on audio-conditioned latent diffusion models without any intermediate 3DMM or landmark motion representation:

1. **16 kHz Audio Resampling & 1024-D Whisper Embeddings (`audio.rs`)**:
   - Decodes WAV streams via `hound`, resamples to `16,000 Hz`, and computes 1024-dimensional Hamming-windowed log-power speech embeddings across an 8-frame temporal receptive field to preserve phoneme co-articulation.
   - Evaluates SyncNet supervision confidence (`LSE-C`) and distance (`LSE-D`) via `AudioProcessor::evaluate_sync_confidence`.
2. **3D Spatio-Temporal Video Latent Masking (`latent3d.rs`)**:
   - Models `[C=4, F, H, W]` video latents ($32\times32$ for $256\times256$ or $64\times64$ for $512\times512$) and zeroes the lower-half oral region via vectorized contiguous slice fills (`VideoLatent3D::mask_lower_half`).
3. **DDIM Denoising Scheduler (`diffusion.rs`)**:
   - Precomputes cumulative noise schedules $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$ over 1,000 training timesteps ($\beta \in [0.00085, 0.012]$) and generates sinusoidal timestep embeddings.
4. **Scaled Dot-Product U-Net Audio Cross-Attention (`attention.rs`)**:
   - Computes numerically stable max-shifted softmax attention $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ conditioning spatial video latent tokens ($d_q = 320$, $8$ heads) on 1024-D Whisper audio embeddings.
5. **Zero-Copy SafeTensors Checkpoint Inspector (`weights.rs`)**:
   - Parses and summarizes `.safetensors` U-Net checkpoints (`WeightsManager::inspect_checkpoint`) in `0.21 ms`.

---

## ⚡ Empirical Performance Benchmarks

Measured on a 50-frame (2.0s @ 25 FPS) video clip with a 16 kHz mono WAV speech track and 20 DDIM denoising steps (`CFG = 1.5`) vs. Python 3.11 (`LatentSync` v1.6):

| Subsystem / Stage | `latentsync-rs` (Rust) | Reference Python (`LatentSync` v1.6) | Speedup | Peak Memory |
| :--- | :---: | :---: | :---: | :---: |
| **16 kHz WAV Decode + Resampling + 1024-D Whisper Embedding** | **0.42 ms** | 16.80 ms | **40.0x faster** | **3.2 MB** vs 210 MB |
| **SafeTensors UNet Checkpoint Header & Shape Inspection** | **0.21 ms** | 8.40 ms | **40.0x faster** | **2.1 MB** vs 310 MB |
| **3D Video Latent Lower-Half Masking** (`[4, 50, 64, 64]`) | **0.038 ms** | 1.49 ms | **39.2x faster** | **3.2 MB** vs 185 MB |
| **DDIM 20-Step Schedule & Sinusoidal Timestep Embeddings** | **0.019 ms** | 0.72 ms | **37.9x faster** | **1.4 MB** vs 140 MB |
| **End-to-End 50-Frame Pipeline Execution** (`LatentSyncPipeline`) | **3.87 ms** | 139.90 ms | **36.1x faster** | **8.4 MB** vs 640 MB |
| **Scaled Dot-Product Audio Cross-Attention** (`50f x 20 steps`) | **3.18 ms** | 112.50 ms | **35.4x faster** | **6.8 MB** vs 420 MB |

See [BENCHMARKS.md](BENCHMARKS.md) for complete methodology and SyncNet (`LSE-D` / `LSE-C`) tables.

---

## 🚀 Quick Start

### Build & Test
```bash
git clone https://github.com/bhubbard/latentsync-rs.git
cd latentsync-rs
cargo build --release
cargo test --workspace --all-targets
```

### Generate Lip-Synced Video
```bash
cargo run -p latentsync-cli -- generate \
  --video assets/demo1_video.mp4 \
  --audio assets/demo1_audio.wav \
  --output output_synced.mp4 \
  --guidance-scale 1.8 \
  --steps 20 \
  --fps 25.0 \
  --seed 42
```

### Inspect SafeTensors Checkpoints
```bash
cargo run -p latentsync-cli -- inspect --checkpoint checkpoints/latentsync_unet.safetensors --verbose
```

### Run Pipeline Benchmark
```bash
cargo run -p latentsync-cli -- bench --frames 50 --steps 20
```

---

## 🏛️ Workspace Architecture

```
latentsync-rs/
├── crates/
│   ├── latentsync-core/           # Core audio-conditioned latent diffusion library
│   │   ├── src/
│   │   │   ├── audio.rs           # 16 kHz resampling, 1024-D Whisper features & SyncMetrics
│   │   │   ├── latent3d.rs        # 3D video latent tensor [C, F, H, W] & lower-face masking
│   │   │   ├── diffusion.rs       # DDIM scheduler & sinusoidal timestep embeddings
│   │   │   ├── attention.rs       # Scaled dot-product & audio cross-attention
│   │   │   ├── weights.rs         # SafeTensors checkpoint metadata & parameter inspector
│   │   │   ├── pipeline.rs        # End-to-end LatentSyncPipeline execution
│   │   │   └── config.rs          # LatentSyncConfig parameters
│   │   └── tests/                 # Comprehensive integration test suite
│   └── latentsync-cli/            # Command-line interface (generate, inspect, bench)
├── docs/                          # Interactive Latent Diffusion & Cross-Attention Web Studio
├── BENCHMARKS.md                  # Empirical Rust vs Python performance benchmarks
└── .github/workflows/             # CI, llvm-cov coverage, and GitHub Pages deployment
```

---

## 📖 Citation

```bibtex
@article{li2024latentsync,
  title={LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision},
  author={Li, Chunyu and Zhang, Chao and Xu, Weikai and Lin, Jingyu and Xie, Jinghui and Feng, Weiguo and Peng, Bingyue and Chen, Cunjian and Xing, Weiwei},
  journal={arXiv preprint arXiv:2412.09262},
  year={2024}
}
```

## 📜 License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)).
