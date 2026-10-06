use serde::{Deserialize, Serialize};

/// Attention weights and dimensions for LatentSync UNet modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub query_dim: usize,
    pub cross_attention_dim: Option<usize>,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            query_dim: 320,
            cross_attention_dim: Some(1024), // Whisper embedding dimension
            num_heads: 8,
            head_dim: 40,
        }
    }
}

pub struct Attention;

impl Attention {
    /// Compute scaled dot-product attention
    /// queries: [B * heads, seq_q, head_dim]
    /// keys:    [B * heads, seq_k, head_dim]
    /// values:  [B * heads, seq_k, head_dim]
    pub fn scaled_dot_product(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_q: usize,
        seq_k: usize,
        dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (dim as f32).sqrt();
        let mut out = vec![0.0f32; seq_q * dim];

        // For each query token
        for i in 0..seq_q {
            let q_slice = &q[i * dim..(i + 1) * dim];

            // Compute attention scores against all key tokens
            let mut scores = Vec::with_capacity(seq_k);
            let mut max_score = f32::NEG_INFINITY;

            for j in 0..seq_k {
                let k_slice = &k[j * dim..(j + 1) * dim];
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += q_slice[d] * k_slice[d];
                }
                dot *= scale;
                if dot > max_score {
                    max_score = dot;
                }
                scores.push(dot);
            }

            // Stable Softmax
            let mut sum_exp = 0.0f32;
            let mut exp_scores = Vec::with_capacity(seq_k);
            for s in scores {
                let exp_s = (s - max_score).exp();
                sum_exp += exp_s;
                exp_scores.push(exp_s);
            }

            if sum_exp > 0.0 {
                for exp_s in &mut exp_scores {
                    *exp_s /= sum_exp;
                }
            }

            // Weighted sum of values
            let out_slice = &mut out[i * dim..(i + 1) * dim];
            for (j, weight) in exp_scores.into_iter().enumerate() {
                let v_slice = &v[j * dim..(j + 1) * dim];
                for d in 0..dim {
                    out_slice[d] += weight * v_slice[d];
                }
            }
        }

        out
    }

    /// Audio cross-attention: Query is video tokens, Key/Value is audio feature embeddings
    pub fn audio_cross_attention(
        video_tokens: &[f32],
        audio_embedding: &[f32],
        num_video_tokens: usize,
        token_dim: usize,
    ) -> Vec<f32> {
        // Project audio embedding to token_dim if needed
        let mut projected_audio = vec![0.0f32; token_dim];
        let audio_dim = audio_embedding.len();
        if audio_dim > 0 {
            for i in 0..token_dim {
                projected_audio[i] = audio_embedding[i % audio_dim];
            }
        }

        Self::scaled_dot_product(
            video_tokens,
            &projected_audio,
            &projected_audio,
            num_video_tokens,
            1,
            token_dim,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_dot_product() {
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![0.5, 0.5, 0.2, 0.8];
        let out = Attention::scaled_dot_product(&q, &k, &v, 2, 2, 2);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_audio_cross_attention() {
        let video = vec![0.1; 16 * 64];
        let audio = vec![0.5; 1024];
        let out = Attention::audio_cross_attention(&video, &audio, 16, 64);
        assert_eq!(out.len(), 16 * 64);
    }
}
