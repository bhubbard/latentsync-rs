use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentSyncConfig {
    /// Classifier-Free Guidance (CFG) audio scale (default 1.5)
    pub guidance_scale: f32,
    /// Diffusion denoising steps (default 20)
    pub num_inference_steps: usize,
    /// Processing video frame rate (default 25.0)
    pub fps: f32,
    /// Random seed for diffusion sampling
    pub seed: u64,
    /// Video latent spatial size (default 256x256 -> 32x32 latent)
    pub size: u32,
    /// Checkpoint directory override
    pub checkpoint_dir: Option<PathBuf>,
}

impl Default for LatentSyncConfig {
    fn default() -> Self {
        Self {
            guidance_scale: 1.5,
            num_inference_steps: 20,
            fps: 25.0,
            seed: 42,
            size: 256,
            checkpoint_dir: None,
        }
    }
}
