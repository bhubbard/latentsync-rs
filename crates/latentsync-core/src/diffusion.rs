use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionStep {
    pub step_idx: usize,
    pub timestep: usize,
    pub alpha_cumprod: f32,
    pub sigma: f32,
}

pub struct DdimScheduler {
    pub num_train_timesteps: usize,
    pub betas: Vec<f32>,
    pub alphas_cumprod: Vec<f32>,
}

impl DdimScheduler {
    pub fn new(num_train_timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        let mut betas = Vec::with_capacity(num_train_timesteps);
        for i in 0..num_train_timesteps {
            let t = i as f32 / (num_train_timesteps - 1) as f32;
            betas.push(beta_start + t * (beta_end - beta_start));
        }

        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut prod = 1.0f32;
        for &b in &betas {
            prod *= 1.0 - b;
            alphas_cumprod.push(prod);
        }

        Self {
            num_train_timesteps,
            betas,
            alphas_cumprod,
        }
    }

    /// Compute inference timesteps schedule (e.g. 20 steps evenly spaced over 1000)
    pub fn get_inference_schedule(&self, num_inference_steps: usize) -> Vec<DiffusionStep> {
        let step_ratio = self.num_train_timesteps / num_inference_steps;
        let mut schedule = Vec::with_capacity(num_inference_steps);

        for step in 0..num_inference_steps {
            let t = (num_inference_steps - 1 - step) * step_ratio;
            let alpha = self.alphas_cumprod[t];
            let sigma = (1.0 - alpha).sqrt();

            schedule.push(DiffusionStep {
                step_idx: step,
                timestep: t,
                alpha_cumprod: alpha,
                sigma,
            });
        }

        schedule
    }

    /// Compute sinusoidal timestep embedding
    pub fn get_timestep_embedding(timestep: usize, dim: usize) -> Vec<f32> {
        let mut emb = vec![0.0f32; dim];
        let half = dim / 2;
        let freq_factor = -(10000.0f32.ln()) / (half as f32 - 1.0);

        for i in 0..half {
            let freq = (i as f32 * freq_factor).exp();
            let arg = timestep as f32 * freq;
            emb[i] = arg.sin();
            emb[half + i] = arg.cos();
        }

        emb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddim_schedule() {
        let scheduler = DdimScheduler::new(1000, 0.00085, 0.012);
        let steps = scheduler.get_inference_schedule(20);
        assert_eq!(steps.len(), 20);
        assert!(steps[0].timestep > steps[19].timestep);
    }
}
