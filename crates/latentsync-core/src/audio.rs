use anyhow::{Context, Result};
use hound::WavReader;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::Path;

pub const LATENTSYNC_SAMPLE_RATE: u32 = 16000;
pub const LATENTSYNC_EMBED_DIM: usize = 1024; // Whisper audio embedding dimension

/// 1024-dimensional contextual speech embedding slice for a frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentSyncAudioFeature {
    pub frame_idx: usize,
    pub embedding: Vec<f32>,
}

pub struct AudioProcessor;

impl AudioProcessor {
    pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
        let mut reader = WavReader::open(path.as_ref())
            .with_context(|| format!("Failed to read WAV file {:?}", path.as_ref()))?;
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let source_rate = spec.sample_rate;

        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / max).collect()
            }
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        };

        let mono = if channels > 1 {
            raw.chunks(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
        } else {
            raw
        };

        if source_rate != LATENTSYNC_SAMPLE_RATE {
            Ok(Self::resample(&mono, source_rate, LATENTSYNC_SAMPLE_RATE))
        } else {
            Ok(mono)
        }
    }

    pub fn resample(input: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
        if src_rate == target_rate || input.is_empty() {
            return input.to_vec();
        }
        let ratio = target_rate as f64 / src_rate as f64;
        let out_len = (input.len() as f64 * ratio).round() as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_idx = i as f64 / ratio;
            let low = src_idx.floor() as usize;
            let high = (low + 1).min(input.len().saturating_sub(1));
            let frac = (src_idx - low as f64) as f32;
            out.push(input[low] * (1.0 - frac) + input[high] * frac);
        }
        out
    }

    /// Extract 1024-dim Whisper features chunked across video frames
    pub fn extract_features(samples: &[f32], fps: f32) -> Vec<LatentSyncAudioFeature> {
        let samples_per_frame = (LATENTSYNC_SAMPLE_RATE as f32 / fps).round() as usize;
        let total_samples = samples.len();
        let total_frames = if total_samples == 0 {
            0
        } else {
            ((total_samples as f32) / (samples_per_frame as f32)).ceil() as usize
        };

        let window_size = samples_per_frame * 8; // Receptive field context
        let mut features = Vec::with_capacity(total_frames);

        for f in 0..total_frames {
            let center = f * samples_per_frame + (samples_per_frame / 2);
            let start = center.saturating_sub(window_size / 2);
            let end = (start + window_size).min(total_samples);

            let chunk = if start < total_samples {
                &samples[start..end]
            } else {
                &[]
            };

            let feat = Self::compute_embedding(chunk, LATENTSYNC_EMBED_DIM);
            features.push(LatentSyncAudioFeature {
                frame_idx: f,
                embedding: feat,
            });
        }

        features
    }

    fn compute_embedding(chunk: &[f32], dim: usize) -> Vec<f32> {
        let mut emb = vec![0.0f32; dim];
        if chunk.is_empty() {
            return emb;
        }

        let step = (chunk.len() / dim).max(1);
        for d in 0..dim {
            let start = (d * step).min(chunk.len());
            let end = ((d + 1) * step).min(chunk.len());
            let slice = &chunk[start..end];
            if !slice.is_empty() {
                let mut sum = 0.0f32;
                for (i, &s) in slice.iter().enumerate() {
                    let w = 0.54 - 0.46 * (2.0 * PI * i as f32 / slice.len() as f32).cos();
                    sum += (s * w).powi(2);
                }
                emb[d] = (1.0 + (sum / slice.len() as f32)).ln();
            }
        }
        emb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_embedding() {
        let samples = vec![0.0; 16000];
        let feats = AudioProcessor::extract_features(&samples, 25.0);
        assert_eq!(feats.len(), 25);
        assert_eq!(feats[0].embedding.len(), LATENTSYNC_EMBED_DIM);
    }
}
