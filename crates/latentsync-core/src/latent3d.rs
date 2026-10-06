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
        for c in 0..self.channels {
            for f in 0..self.frames {
                for h in mid_h..self.height {
                    for w in 0..self.width {
                        let idx = ((c * self.frames + f) * self.height + h) * self.width + w;
                        if idx < self.data.len() {
                            self.data[idx] = 0.0;
                        }
                    }
                }
            }
        }
    }
}
