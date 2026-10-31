use latentsync_core::{
    Attention, AttentionConfig, AudioProcessor, DdimScheduler, LatentSyncConfig,
    LatentSyncPipeline, VideoLatent3D, WeightsManager, LATENTSYNC_EMBED_DIM, LATENTSYNC_SAMPLE_RATE,
};
use tempfile::NamedTempFile;

#[test]
fn test_audio_constants_and_dimensions() {
    assert_eq!(LATENTSYNC_SAMPLE_RATE, 16000);
    assert_eq!(LATENTSYNC_EMBED_DIM, 1024);

    let dummy_audio = vec![0.1f32; 16000 * 2]; // 2 seconds
    let features = AudioProcessor::extract_features(&dummy_audio, 25.0);
    assert_eq!(features.len(), 50); // 25 fps * 2s
    assert_eq!(features[0].embedding.len(), LATENTSYNC_EMBED_DIM);
}

#[test]
fn test_ddim_scheduler_timesteps() {
    let scheduler = DdimScheduler::new(1000, 0.00085, 0.012);
    assert_eq!(scheduler.num_train_timesteps, 1000);
    assert_eq!(scheduler.alphas_cumprod.len(), 1000);

    let steps = scheduler.get_inference_schedule(20);
    assert_eq!(steps.len(), 20);

    // Timesteps should be descending towards 0
    assert!(steps[0].timestep > steps[19].timestep);

    // Alpha values should be within [0.0, 1.0]
    for &alpha in &scheduler.alphas_cumprod {
        assert!((0.0..=1.0).contains(&alpha));
    }
}

#[test]
fn test_attention_mechanism() {
    let q = vec![1.0f32; 64];
    let k = vec![1.0f32; 64];
    let v = vec![0.5f32; 64];

    let out = Attention::scaled_dot_product(&q, &k, &v, 1, 1, 64);
    assert_eq!(out.len(), 64);

    let default_cfg = AttentionConfig::default();
    assert_eq!(default_cfg.query_dim, 320);
    assert_eq!(default_cfg.cross_attention_dim, Some(1024));
}

#[test]
fn test_latent3d_dimensions_and_masking() {
    // 256x256 image with VAE 8x downsampling -> 32x32 latent
    let mut latent = VideoLatent3D::new(4, 16, 32, 32);
    assert_eq!(latent.channels, 4);
    assert_eq!(latent.frames, 16);
    assert_eq!(latent.height, 32);
    assert_eq!(latent.width, 32);
    assert_eq!(latent.data.len(), 4 * 16 * 32 * 32);

    // Fill with 1.0 then mask
    latent.data.fill(1.0);
    latent.mask_lower_half();

    // Top row (h=0) unmasked
    assert_eq!(latent.data[0], 1.0);
    // Bottom row (h=20) masked
    let bottom_idx = 20 * 32;
    assert_eq!(latent.data[bottom_idx], 0.0);
}

#[test]
fn test_pipeline_execution() {
    let temp_video = NamedTempFile::new().unwrap();
    let temp_audio = NamedTempFile::new().unwrap();
    let temp_out = NamedTempFile::new().unwrap();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(temp_audio.path(), spec).unwrap();
    for _ in 0..16000 {
        writer.write_sample(0i16).unwrap();
    }
    writer.finalize().unwrap();

    let config = LatentSyncConfig {
        num_inference_steps: 2,
        guidance_scale: 1.5,
        fps: 25.0,
        ..Default::default()
    };
    let pipeline = LatentSyncPipeline::new(config);
    let result = pipeline.run(temp_video.path(), temp_audio.path(), temp_out.path()).unwrap();
    assert!(result.total_frames > 0);
    assert_eq!(result.diffusion_steps, 2);
    assert_eq!(result.guidance_scale, 1.5);
}

#[test]
fn test_weights_manager_invalid_path() {
    let summary = WeightsManager::inspect_checkpoint("/invalid/latent_sync.safetensors");
    assert!(summary.is_err());
}
