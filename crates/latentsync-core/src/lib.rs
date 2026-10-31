//! latentsync-core: High-performance audio-conditioned latent diffusion engine for video lip-sync.

pub mod attention;
pub mod audio;
pub mod config;
pub mod diffusion;
pub mod latent3d;
pub mod pipeline;
pub mod weights;

pub use attention::{Attention, AttentionConfig};
pub use audio::{
    AudioProcessor, LatentSyncAudioFeature, SyncMetrics, LATENTSYNC_EMBED_DIM,
    LATENTSYNC_SAMPLE_RATE,
};
pub use config::LatentSyncConfig;
pub use diffusion::{DdimScheduler, DiffusionStep};
pub use latent3d::VideoLatent3D;
pub use pipeline::{LatentSyncPipeline, PipelineExecutionResult};
pub use weights::{CheckpointSummary, TensorInfo, WeightsManager};
