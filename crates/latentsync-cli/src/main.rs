use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use latentsync_core::{
    LatentSyncConfig, LatentSyncPipeline, WeightsManager,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "latentsync",
    version,
    about = "High-performance CLI for LatentSync: Audio-Conditioned Latent Diffusion for Video Lip Sync",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate synchronized video from input video and speech audio
    Generate {
        /// Path to input target video or image
        #[arg(short = 'v', long = "video")]
        video: PathBuf,

        /// Path to driving audio file (.wav, .mp3)
        #[arg(short = 'a', long = "audio")]
        audio: PathBuf,

        /// Path to output synchronized video file (.mp4)
        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        /// Classifier-Free Guidance (CFG) scale
        #[arg(long = "guidance-scale", default_value = "1.5")]
        guidance_scale: f32,

        /// Number of DDIM diffusion denoising steps
        #[arg(long = "steps", default_value = "20")]
        steps: usize,

        /// Video frame rate
        #[arg(long = "fps", default_value = "25.0")]
        fps: f32,

        /// Random seed for sampling reproducibility
        #[arg(long = "seed", default_value = "42")]
        seed: u64,

        /// Optional directory containing pretrained model checkpoints
        #[arg(long = "checkpoint-dir")]
        checkpoint_dir: Option<PathBuf>,
    },

    /// Inspect a SafeTensors checkpoint or model weights
    Inspect {
        /// Path to the SafeTensors (.safetensors) checkpoint
        #[arg(short = 'c', long = "checkpoint")]
        checkpoint: PathBuf,

        /// Display detailed tensor lists
        #[arg(short = 'v', long = "verbose")]
        verbose: bool,
    },

    /// Run synthetic diffusion pipeline benchmark
    Bench {
        /// Number of frames to benchmark
        #[arg(long = "frames", default_value = "50")]
        frames: usize,

        /// Number of diffusion steps
        #[arg(long = "steps", default_value = "20")]
        steps: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            video,
            audio,
            output,
            guidance_scale,
            steps,
            fps,
            seed,
            checkpoint_dir,
        } => {
            println!(
                "{} LatentSync audio-conditioned latent diffusion synthesis...",
                "Starting".cyan().bold()
            );
            println!("  Input Video : {}", video.display().to_string().yellow());
            println!("  Input Audio : {}", audio.display().to_string().yellow());
            println!("  Output File : {}", output.display().to_string().green());
            println!("  CFG Scale   : {}", guidance_scale);
            println!("  DDIM Steps  : {}", steps);

            let pb = ProgressBar::new(steps as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} steps ({eta})")?
                    .progress_chars("#>-"),
            );

            let config = LatentSyncConfig {
                guidance_scale,
                num_inference_steps: steps,
                fps,
                seed,
                size: 256,
                checkpoint_dir,
            };

            let pipeline = LatentSyncPipeline::new(config);
            let result = pipeline.run(&video, &audio, &output)
                .with_context(|| "Pipeline synthesis execution failed")?;

            pb.finish_with_message("Done!");

            println!("\n{}", "Synthesis complete!".green().bold());
            println!("  Total Frames     : {}", result.total_frames);
            println!("  Duration         : {:.2}s", result.duration_seconds);
            println!("  Output Saved To  : {}", result.output_path.display().to_string().green());
        }

        Commands::Inspect { checkpoint, verbose } => {
            println!(
                "{} SafeTensors checkpoint: {}",
                "Inspecting".cyan().bold(),
                checkpoint.display()
            );

            let summary = WeightsManager::inspect_checkpoint(&checkpoint)?;
            println!("\n{}", "Checkpoint Summary:".bold());
            println!("  Total Tensors   : {}", summary.tensor_count.to_string().cyan());
            println!("  Total Parameters: {}", format!("{:e}", summary.total_parameters as f64).yellow());

            if verbose {
                println!("\n{}", "Tensor Metadata:".bold());
                for t in &summary.tensors {
                    println!("  - {:<40} shape: {:?}, dtype: {}", t.name, t.shape, t.dtype);
                }
            }
        }

        Commands::Bench { frames, steps } => {
            println!(
                "{} LatentSync benchmark: {} frames, {} steps...",
                "Running".cyan().bold(),
                frames,
                steps
            );

            let start = Instant::now();
            let scheduler = latentsync_core::DdimScheduler::new(1000, 0.00085, 0.012);
            let schedule = scheduler.get_inference_schedule(steps);
            let mut latent = latentsync_core::VideoLatent3D::new(4, frames, 32, 32);
            latent.mask_lower_half();

            for _step in &schedule {
                for _f in 0..frames {
                    let _emb = latentsync_core::Attention::audio_cross_attention(&vec![0.1; 1024], &vec![0.5; 1024], 16, 64);
                }
            }

            let elapsed = start.elapsed();
            let fps = frames as f64 / elapsed.as_secs_f64();

            println!("\n{}", "Benchmark Results:".green().bold());
            println!("  Time Elapsed    : {:.3}s", elapsed.as_secs_f64());
            println!("  Processing Speed: {:.2} fps", fps);
        }
    }

    Ok(())
}
