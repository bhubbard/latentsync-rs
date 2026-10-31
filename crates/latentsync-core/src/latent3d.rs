use serde::{Deserialize, Serialize};

/// 3D Video Latent Tensor [Channels x Frames x Height x Width]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoLatent3D {
    pub channels: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl VideoLatent3D {
    pub fn new(channels: usize, frames: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            frames,
            height,
            width,
            data: vec![0.0f32; channels * frames * height * width],
        }
    }

    /// Zero out lower face area across all frames in video latent
    pub fn mask_lower_half(&mut self) {
        let mid_h = self.height / 2;
        let frame_stride = self.height * self.width;
        let mask_start_offset = mid_h * self.width;
        for c in 0..self.channels {
            for f in 0..self.frames {
                let frame_base = (c * self.frames + f) * frame_stride;
                let start = (frame_base + mask_start_offset).min(self.data.len());
                let end = (frame_base + frame_stride).min(self.data.len());
                self.data[start..end].fill(0.0);
            }
        }
    }
}
