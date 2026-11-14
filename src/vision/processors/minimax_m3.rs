//! MiniMax-M3 vision processor.
//!
//! Despite the `image_grid_pinpoints` and `process_image_mode: "dynamic_res"`
//! keys in its config — which read as LLaVA-NeXT tiling — MiniMax-M3 preprocesses
//! images the Qwen2-VL way: a resize onto a `patch_size * merge_size` grid,
//! patchified into a flat
//! `[total_patches, channels * temporal_patch_size * patch_size^2]` tensor
//! alongside an `image_grid_thw` triple. The engine's M3 vision tower consumes exactly
//! that layout, so this processor wraps the shared [`QwenVLProcessorBase`]
//! and supplies M3's own parameters.
//!
//! # Resize policy
//!
//! The resize rule is configurable; see [`QwenResizePolicy`]. The default is
//! the MiniMax-M3.1 **aligned long-side resize**
//! ([`QwenResizePolicy::AlignedLongSide`]): inputs are floored to an area of
//! 168² and a short side of 168, both sides round UP to the 28 px factor, and
//! each side is capped at a per-modality long side — for images the
//! OpenAI-style `detail` tier (low 672 / auto 2,016 / high 3,584, default
//! auto; an explicit `max_long_side_pixel` wins, clamped to 3,584), for video
//! frames the video tier (low 504 / auto 672 / high 2,188; a request's
//! `max_long_side_pixel` wins). A clip whose resized
//! `width * height * frames` exceeds 301,056,000 is rejected. There is no
//! area cap.
//!
//! `"resize_policy": "smart_resize"` in the preprocessor config selects the
//! previous behavior instead: Qwen2-VL's `smart_resize` with M3's bounds
//! (min_pixels 3,136, max_pixels 451,584 = `image_seq_length` 576 tokens,
//! video frames bounded at 602,112 px with `max_long_side_pixel` tiers),
//! including the 112 px short-side raise. Every aligned number, and the
//! policy itself, is overridable from the preprocessor config — see the
//! [`resize_policy`](super::resize_policy) module docs for the keys.
//!
//! # MiniMax-M3 parameters
//!
//! - patch_size: 14
//! - merge_size: 2 (`img_token_compression_config.spatial_merge_size`)
//! - temporal_patch_size: 2 (`img_token_compression_config.temporal_patch_size`)
//! - factor: 28 (patch_size * merge_size)
//! - normalization: CLIP mean/std
//!
//! The smart-resize bounds differ from Qwen2-VL's (200,704 / 1,003,520), so
//! M3 cannot simply reuse the Qwen2-VL processor's defaults.

use std::{borrow::Cow, ops::Deref};

use image::{imageops::FilterType, DynamicImage};

use super::{
    qwen2_vl::{CLIP_MEAN, CLIP_STD},
    qwen_vl_base::{QwenVLConfig, QwenVLProcessorBase, QwenVideoResizeMode},
    resize_policy::{self, QwenResizePolicy},
};
use crate::{
    types::RgbFrameRef,
    vision::{
        preprocessor_config::PreProcessorConfig,
        processor::{
            PreprocessedEncoderInputs, VideoClipOptions, VisionPreProcessor,
            VisionPreprocessingContext,
        },
        transforms::{resize, resize_bicubic_torch_aa, TransformError},
    },
};

/// Default patch size.
pub const DEFAULT_PATCH_SIZE: usize = 14;

/// Default spatial merge size (2x2 patch merge before the projector).
pub const DEFAULT_MERGE_SIZE: usize = 2;

/// Default temporal patch size (video frames are padded to a multiple of this).
pub const DEFAULT_TEMPORAL_PATCH_SIZE: usize = 2;

/// Default minimum pixels (4 * 28 * 28 = 3,136), sizing the smart resize
/// policy; the aligned policy floors the area at 168² instead.
pub const DEFAULT_MIN_PIXELS: usize = 4 * 28 * 28;

/// Default maximum pixels for images (576 * 28 * 28 = 451,584), the smart
/// resize policy's area cap.
///
/// 576 is the model's `image_seq_length`: at the bound, an image occupies
/// exactly `image_seq_length` tokens after the 2x2 merge (2,304 patches before
/// it).
pub const DEFAULT_MAX_PIXELS: usize = 576 * 28 * 28;

/// Default maximum pixels per video frame (768 * 28 * 28 = 602,112) under the
/// smart resize policy, not per sampled volume.
pub const DEFAULT_VIDEO_MAX_PIXELS: usize = 768 * 28 * 28;

/// Short side floor in pixels of the smart resize policy's staging raise,
/// four patch factors; smaller images are scaled up to it first.
const MIN_SHORT_SIDE: u32 = 112;

/// The base's aspect-ratio guard (`smart_resize` in qwen_vl_base.rs), which a staged image must
/// satisfy too; keep the two in step.
const MAX_ASPECT_RATIO: f64 = 200.0;

/// How far over the pixel budget a shrunk staging image lands, so the base still scales it down.
const STAGING_OVERSHOOT: f64 = 1.25;

/// The config block holding M3's merge parameters.
const COMPRESSION_CONFIG_KEY: &str = "img_token_compression_config";

/// MiniMax-M3 image/video processor.
#[derive(Clone)]
pub struct MiniMaxM3VisionProcessor {
    inner: QwenVLProcessorBase,
    /// Whether the config set `max_pixels`, which then bounds video frames
    /// in place of a request's `max_long_side_pixel` tier.
    explicit_max_pixels: bool,
}

impl Default for MiniMaxM3VisionProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl MiniMaxM3VisionProcessor {
    /// Create a processor with MiniMax-M3's default parameters and the M3.1
    /// aligned long-side resize policy.
    pub fn new() -> Self {
        Self::build(
            DEFAULT_PATCH_SIZE,
            DEFAULT_MERGE_SIZE,
            DEFAULT_TEMPORAL_PATCH_SIZE,
            DEFAULT_MIN_PIXELS,
            DEFAULT_MAX_PIXELS,
            DEFAULT_VIDEO_MAX_PIXELS,
            QwenResizePolicy::AlignedLongSide(resize_policy::minimax_m31_aligned()),
        )
    }

    /// Create a processor with custom smart-resize settings.
    ///
    /// The bounds this constructor takes are the smart resize rule's, so it
    /// selects [`QwenResizePolicy::SmartResize`]; a preprocessor config can
    /// still switch the policy per request layering.
    pub fn with_config(
        patch_size: usize,
        merge_size: usize,
        min_pixels: usize,
        max_pixels: usize,
        temporal_patch_size: usize,
    ) -> Self {
        // The caller's `max_pixels` governs video too; flooring it at the
        // video default would ignore a deliberate reduction.
        Self::build(
            patch_size,
            merge_size,
            temporal_patch_size,
            min_pixels,
            max_pixels,
            max_pixels,
            QwenResizePolicy::SmartResize,
        )
    }

    fn build(
        patch_size: usize,
        merge_size: usize,
        temporal_patch_size: usize,
        min_pixels: usize,
        max_pixels: usize,
        video_max_pixels: usize,
        resize_policy: QwenResizePolicy,
    ) -> Self {
        Self {
            inner: QwenVLProcessorBase::new(QwenVLConfig {
                patch_size,
                merge_size,
                min_pixels,
                max_pixels,
                video_min_pixels: min_pixels,
                video_max_pixels,
                video_resize_mode: QwenVideoResizeMode::PerFrame,
                resize_policy,
                temporal_patch_size,
                mean: CLIP_MEAN,
                std: CLIP_STD,
                model_name: "minimax_m3",
            })
            .allow_video_dimensions_below_factor()
            .raise_rounded_sides_to_factor(),
            explicit_max_pixels: false,
        }
    }

    /// Read a `usize` out of M3's `img_token_compression_config` block.
    ///
    /// M3 nests its merge parameters there rather than exposing the flat
    /// `merge_size` / `temporal_patch_size` keys Qwen models use, so they land
    /// in `PreProcessorConfig::extra` instead of the typed fields.
    ///
    /// Returns `Ok(None)` only when the key is genuinely absent. A present but
    /// malformed value (non-integer, or zero) is an error rather than a silent
    /// fallback: defaulting there would run every request with tensor geometry
    /// that disagrees with the checkpoint.
    fn compression_usize(
        config: &PreProcessorConfig,
        key: &str,
    ) -> Result<Option<usize>, TransformError> {
        let Some(block) = config.extra.get(COMPRESSION_CONFIG_KEY) else {
            return Ok(None);
        };
        let Some(value) = block.get(key) else {
            return Ok(None);
        };
        match value.as_u64() {
            Some(parsed) if parsed > 0 => Ok(Some(parsed as usize)),
            _ => Err(TransformError::ShapeError(format!(
                "minimax_m3: {COMPRESSION_CONFIG_KEY}.{key} must be a positive integer, got {value}"
            ))),
        }
    }

    /// The `temporal_patch_size` the config declares: a positive flat key, else the compression block.
    fn declared_temporal_patch_size(
        config: &PreProcessorConfig,
    ) -> Result<Option<usize>, TransformError> {
        let nested = Self::compression_usize(config, "temporal_patch_size")?;
        Ok(config
            .temporal_patch_size
            .filter(|&size| size > 0)
            .or(nested))
    }

    /// Frames per temporal patch this processor pairs under `config`: the flat
    /// key, else the compression block, else the default. A present but
    /// unusable block value is the configuration error the construction path
    /// reports, so a prompt's stamps are never paired by a default that the
    /// preprocessing of the same config refuses.
    pub fn temporal_patch_size_from(config: &PreProcessorConfig) -> Result<usize, TransformError> {
        Ok(Self::declared_temporal_patch_size(config)?.unwrap_or(DEFAULT_TEMPORAL_PATCH_SIZE))
    }

    /// Build a processor from a preprocessor config, falling back to M3's
    /// defaults for anything the config does not specify.
    ///
    /// A present but malformed `img_token_compression_config` value fails
    /// here, at construction, rather than silently falling back to the
    /// defaults and hiding the real cause until the first request.
    pub fn from_preprocessor_config(config: &PreProcessorConfig) -> Result<Self, TransformError> {
        Self::new().layered_over(config)
    }

    /// Layer a config over this processor's settings.
    ///
    /// Values the config does not specify keep whatever this processor was
    /// built with, so settings supplied through [`Self::with_config`] survive
    /// into `preprocess` and `calculate_num_tokens` instead of being reset to
    /// the checkpoint defaults.
    fn layered_over(&self, config: &PreProcessorConfig) -> Result<Self, TransformError> {
        let merge_size = config
            .merge_size
            .or(Self::compression_usize(config, "spatial_merge_size")?)
            .unwrap_or_else(|| self.inner.merge_size());
        let temporal_patch_size = Self::declared_temporal_patch_size(config)?
            .unwrap_or_else(|| self.inner.temporal_patch_size());
        let patch_size = config.get_patch_size(self.inner.patch_size());
        let factor = patch_size * merge_size;

        // The resize policy the config selects, layered over this processor's
        // aligned settings when it already runs the aligned policy (or over
        // the M3.1 defaults when it does not).
        let aligned_base = match self.inner.resize_policy() {
            QwenResizePolicy::AlignedLongSide(aligned) => aligned.clone(),
            QwenResizePolicy::SmartResize => resize_policy::minimax_m31_aligned(),
        };
        let policy = QwenResizePolicy::from_config(config, &aligned_base, factor)?
            .unwrap_or_else(|| self.inner.resize_policy().clone());

        let max_pixels = config.max_pixels.unwrap_or_else(|| self.inner.max_pixels());
        let min_pixels = config.min_pixels.unwrap_or_else(|| self.inner.min_pixels());
        // Track an explicit `max_pixels` in both directions: flooring the video
        // budget at the default would leave video six times an image's budget
        // for an operator who lowered `max_pixels` to bound encoder memory.
        let video_max_pixels = config
            .max_pixels
            .unwrap_or_else(|| self.inner.video_max_pixels());

        let mut layered = Self::build(
            patch_size,
            merge_size,
            temporal_patch_size,
            min_pixels,
            max_pixels,
            video_max_pixels,
            policy,
        );
        layered.inner = layered.inner.with_preprocessor_config(config);
        layered.explicit_max_pixels = self.explicit_max_pixels || config.max_pixels.is_some();
        Ok(layered)
    }

    /// This processor for one clip.
    ///
    /// Under the smart resize policy a `max_long_side_pixel` tier on the
    /// request is the caller's frame budget, unless `max_pixels` was set
    /// explicitly; the budget is the square of the long side, so a frame
    /// already capped to that side is never shrunk again. Under the aligned
    /// policy the tier is the frame long-side cap itself, winning over the
    /// config's detail tier and explicit cap.
    fn for_clip(&self, clip: &VideoClipOptions) -> Cow<'_, Self> {
        match self.inner.resize_policy() {
            QwenResizePolicy::AlignedLongSide(_) => match clip.max_long_side_pixel {
                Some(tier) => {
                    let mut processor = self.clone();
                    processor.inner = processor.inner.with_video_long_side_override(Some(tier));
                    Cow::Owned(processor)
                }
                None => Cow::Borrowed(self),
            },
            QwenResizePolicy::SmartResize => match clip.max_long_side_pixel {
                Some(tier) if !self.explicit_max_pixels => {
                    let tier = tier as usize;
                    let budget = tier.saturating_mul(tier).max(self.inner.min_pixels());
                    let mut processor = self.clone();
                    processor.inner = processor.inner.with_video_max_pixels(budget);
                    Cow::Owned(processor)
                }
                _ => Cow::Borrowed(self),
            },
        }
    }

    /// Dimensions with the short side raised to [`MIN_SHORT_SIDE`]; `None` if at it or zero.
    ///
    /// Part of the smart resize policy's staging; the aligned policy's floors
    /// replace it.
    fn raised_dimensions(width: u32, height: u32) -> Option<(u32, u32)> {
        let short = width.min(height);
        if short == 0 || short >= MIN_SHORT_SIDE {
            return None;
        }
        let long = (f64::from(width.max(height)) * f64::from(MIN_SHORT_SIDE) / f64::from(short))
            .round() as u32;
        Some(if width <= height {
            (MIN_SHORT_SIDE, long)
        } else {
            (long, MIN_SHORT_SIDE)
        })
    }

    /// Dimensions to hand the base for an image below the floor; `None` if it is at the floor.
    ///
    /// Within the pixel budget that is the raised image itself. Above it the
    /// base's own target is staged when the base would hand it back unchanged
    /// (one resample, at most the budget). Otherwise, for a very thin image or
    /// a lowered budget, the raised image is shrunk to just over the budget,
    /// which trades about a percent of grid fidelity for the bound.
    fn staging_dimensions(
        &self,
        width: u32,
        height: u32,
    ) -> Result<Option<(u32, u32)>, TransformError> {
        let Some((raised_w, raised_h)) = Self::raised_dimensions(width, height) else {
            return Ok(None);
        };
        let (target_h, target_w) = self
            .inner
            .smart_resize(raised_h as usize, raised_w as usize)?;
        let budget = self.inner.max_pixels();
        if raised_w as usize * raised_h as usize <= budget {
            return Ok(Some((raised_w, raised_h)));
        }
        let target_fits = target_w * target_h <= budget
            && aspect_ratio(target_w, target_h) <= MAX_ASPECT_RATIO
            && self.inner.smart_resize(target_h, target_w)? == (target_h, target_w);
        if target_fits {
            return Ok(Some((target_w as u32, target_h as u32)));
        }
        let shrink = (f64::from(raised_w) * f64::from(raised_h)
            / (budget as f64 * STAGING_OVERSHOOT))
            .sqrt();
        // Round the short side up and the long side down so the ratio stays inside the guard.
        let shrunk = |side: u32, short: bool| {
            let scaled = f64::from(side) / shrink;
            if short { scaled.ceil() } else { scaled.floor() }.max(1.0) as u32
        };
        let (w, h) = if raised_w <= raised_h {
            (shrunk(raised_w, true), shrunk(raised_h, false))
        } else {
            (shrunk(raised_w, false), shrunk(raised_h, true))
        };
        let over_budget = w as usize * h as usize > budget;
        if over_budget && aspect_ratio(w as usize, h as usize) <= MAX_ASPECT_RATIO {
            Ok(Some((w, h)))
        } else {
            Ok(Some((raised_w, raised_h)))
        }
    }

    /// Stage every image below the floor, or `None` when none needs it so the batch stays borrowed.
    ///
    /// Smart resize policy only; the aligned policy's floors make the raise
    /// redundant. Otherwise the whole batch is copied, not just the staged
    /// images, so the base gets one slice.
    fn raise_images(
        &self,
        images: &[DynamicImage],
    ) -> Result<Option<Vec<DynamicImage>>, TransformError> {
        let below_floor =
            |image: &DynamicImage| Self::raised_dimensions(image.width(), image.height()).is_some();
        if !self.inner.resizes() || !images.iter().any(below_floor) {
            return Ok(None);
        }
        // Vet every image first so an off-contract input is rejected before any pixel work.
        for image in images {
            self.inner
                .smart_resize(image.height() as usize, image.width() as usize)?;
        }
        // The base's grid stage picks its kernel the same way, so both stages use
        // one filter and, for BICUBIC, the same torchvision `uint8` kernel (the
        // reference image processor is torchvision-backed and resizes once; this
        // raise is the port's staging step before that resize).
        let filter = self.inner.filter();
        let mut staged = Vec::with_capacity(images.len());
        for image in images {
            staged.push(
                match self.staging_dimensions(image.width(), image.height())? {
                    Some((width, height)) if filter == FilterType::CatmullRom => {
                        resize_bicubic_torch_aa(image, width, height)
                    }
                    Some((width, height)) => resize(image, width, height, filter),
                    None => image.clone(),
                },
            );
        }
        Ok(Some(staged))
    }
}

/// Long side over short side.
fn aspect_ratio(a: usize, b: usize) -> f64 {
    a.max(b) as f64 / a.min(b).max(1) as f64
}

impl Deref for MiniMaxM3VisionProcessor {
    type Target = QwenVLProcessorBase;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl VisionPreProcessor for MiniMaxM3VisionProcessor {
    fn emits_pixel_bytes(&self) -> bool {
        true
    }

    fn default_mean(&self) -> [f64; 3] {
        self.inner.default_mean()
    }

    fn default_std(&self) -> [f64; 3] {
        self.inner.default_std()
    }

    fn preprocess(
        &self,
        images: &[DynamicImage],
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        // The aligned policy's floors replace the 112 px staging raise.
        if matches!(
            self.inner.resize_policy(),
            QwenResizePolicy::AlignedLongSide(_)
        ) {
            return self.inner.preprocess(images);
        }
        let Some(raised) = self.raise_images(images)? else {
            return self.inner.preprocess(images);
        };
        let mut out = self.inner.preprocess(&raised)?;
        // Report the caller's sizes, not the raised ones.
        out.item_sizes = images
            .iter()
            .map(|image| (image.width(), image.height()))
            .collect();
        Ok(out)
    }

    /// Under the aligned policy a request's image cap is the resize limit
    /// itself, as in the reference (`resolve_max_long_side_pixel`).
    fn applies_image_long_side(&self) -> bool {
        matches!(
            self.inner.resize_policy(),
            QwenResizePolicy::AlignedLongSide(_)
        )
    }

    fn preprocess_with_context(
        &self,
        images: &[DynamicImage],
        context: &VisionPreprocessingContext,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        match context
            .image_long_side
            .filter(|_| self.applies_image_long_side())
        {
            Some(cap) => {
                let mut processor = self.clone();
                processor.inner = processor.inner.with_image_long_side_override(Some(cap));
                processor.preprocess(images)
            }
            None => self.preprocess(images),
        }
    }

    fn preprocess_video(
        &self,
        frames: &[DynamicImage],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.for_clip(clip).inner.preprocess_video(frames, clip)
    }

    fn preprocess_video_rgb(
        &self,
        frames: &[RgbFrameRef<'_>],
        clip: &VideoClipOptions,
    ) -> Result<PreprocessedEncoderInputs, TransformError> {
        self.for_clip(clip).inner.preprocess_video_rgb(frames, clip)
    }

    fn calculate_num_tokens(&self, width: u32, height: u32) -> Result<usize, TransformError> {
        // The aligned policy has no staging; the base's target is the count's.
        if matches!(
            self.inner.resize_policy(),
            QwenResizePolicy::AlignedLongSide(_)
        ) {
            return self.inner.calculate_num_tokens(width, height);
        }
        // Staged the way `preprocess` stages the image.
        let staged = if self.inner.resizes() {
            self.staging_dimensions(width, height)?
        } else {
            None
        };
        let (width, height) = staged.unwrap_or((width, height));
        self.inner.calculate_num_tokens(width, height)
    }

    fn model_name(&self) -> &'static str {
        self.inner.model_name()
    }

    fn get_processed_size(&self) -> Option<(u32, u32)> {
        self.inner.get_processed_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `img_token_compression_config` block as it appears in the
    /// MiniMax-M3 checkpoint's `preprocessor_config.json`.
    fn m3_config() -> PreProcessorConfig {
        serde_json::from_str(
            r#"{
                "processor_class": "MiniMaxVLProcessor",
                "process_image_mode": "dynamic_res",
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "size": [672, 672],
                "patch_size": 14,
                "img_token_compression_config": {
                    "image_token_compression_threshold": 1.1,
                    "image_token_compression_method": "patch_merge",
                    "max_image_resolution": 1008,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2
                },
                "add_start_end_special_tokens": true
            }"#,
        )
        .expect("checkpoint preprocessor config parses")
    }

    /// `m3_config` with the smart resize policy selected: the sizing behavior
    /// before the aligned long-side resize became the default.
    fn smart_config() -> PreProcessorConfig {
        let mut config = m3_config();
        config.extra.insert(
            "resize_policy".to_string(),
            serde_json::json!("smart_resize"),
        );
        config
    }

    fn smart_processor() -> MiniMaxM3VisionProcessor {
        MiniMaxM3VisionProcessor::from_preprocessor_config(&smart_config())
            .expect("smart-resize config is valid")
    }

    /// The checkpoint config with the aligned policy tuned by `policy`.
    fn aligned_config(policy: serde_json::Value) -> PreProcessorConfig {
        let mut config = m3_config();
        config.extra.insert("resize_policy".to_string(), policy);
        config
    }

    #[test]
    fn defaults_match_the_checkpoint() {
        let processor = MiniMaxM3VisionProcessor::new();
        assert_eq!(processor.patch_size(), 14);
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.temporal_patch_size(), 2);
        assert_eq!(processor.min_pixels(), 3136);
        assert_eq!(processor.max_pixels(), 451_584);
        assert_eq!(processor.model_name(), "minimax_m3");
        // The M3.1 aligned long-side resize is the default policy.
        assert!(matches!(
            processor.resize_policy(),
            QwenResizePolicy::AlignedLongSide(_)
        ));
    }

    #[test]
    fn max_pixels_matches_image_seq_length() {
        // 576 is the model's image_seq_length; at the bound an image is
        // exactly that many patches before the 2x2 merge.
        let processor = MiniMaxM3VisionProcessor::new();
        let factor = processor.patch_size() * processor.merge_size();
        assert_eq!(processor.max_pixels() / (factor * factor), 576);
    }

    #[test]
    fn reads_merge_params_from_the_nested_compression_block() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config())
            .expect("checkpoint config is valid");

        // Neither key exists at the top level of M3's config; both must be
        // picked up from img_token_compression_config.
        assert_eq!(processor.merge_size(), 2);
        assert_eq!(processor.temporal_patch_size(), 2);
        assert_eq!(processor.patch_size(), 14);
    }

    #[test]
    fn checkpoint_config_keeps_m3_pixel_bounds() {
        // M3's config carries no min_pixels/max_pixels, so the M3 defaults must
        // survive rather than falling back to Qwen2-VL's much larger bounds.
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config())
            .expect("checkpoint config is valid");
        assert_eq!(processor.min_pixels(), 3136);
        assert_eq!(processor.max_pixels(), 451_584);
    }

    #[test]
    fn explicit_flat_keys_win_over_the_nested_block() {
        let mut config = m3_config();
        config.merge_size = Some(4);
        config.temporal_patch_size = Some(1);

        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config)
            .expect("checkpoint config is valid");
        assert_eq!(processor.merge_size(), 4);
        assert_eq!(processor.temporal_patch_size(), 1);
    }

    #[test]
    fn token_count_follows_the_merged_grid() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();

        // A 448x448 image is 32x32 patches at patch_size 14, which is
        // 16x16 = 256 tokens after the 2x2 merge.
        assert_eq!(processor.calculate_num_tokens(448, 448).unwrap(), 256);
    }

    #[test]
    fn with_config_settings_survive_request_layering() {
        // A processor built with explicit settings must keep them when a
        // request config does not override them.
        let processor = MiniMaxM3VisionProcessor::with_config(14, 2, 3136, 200_704, 2);
        let mut config = m3_config();
        // The checkpoint block carries merge/temporal but no pixel bounds.
        config.max_pixels = None;
        config.min_pixels = None;

        let layered = processor.layered_over(&config).unwrap();
        assert_eq!(layered.max_pixels(), 200_704);
        assert_eq!(layered.min_pixels(), 3136);
    }

    #[test]
    fn lowering_max_pixels_also_lowers_the_video_budget() {
        let processor = smart_processor();
        let mut config = smart_config();
        config.max_pixels = Some(100_352);

        let layered = processor.layered_over(&config).unwrap();
        assert_eq!(layered.max_pixels(), 100_352);
        // Must track downwards, not stay floored at the video default.
        assert_eq!(layered.video_max_pixels(), 100_352);
    }

    #[test]
    fn a_video_tier_sets_the_frame_budget() {
        let processor = smart_processor();
        let clip = VideoClipOptions {
            sample_fps: 2.0,
            max_long_side_pixel: Some(2016),
        };

        let layered = processor.for_clip(&clip);
        assert_eq!(layered.video_max_pixels(), 2016 * 2016);
        // The tier speaks for video frames only.
        assert_eq!(layered.max_pixels(), DEFAULT_MAX_PIXELS);

        // An explicit max_pixels still wins over the tier.
        let mut config = smart_config();
        config.max_pixels = Some(100_352);
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
        let layered = processor.for_clip(&clip);
        assert_eq!(layered.video_max_pixels(), 100_352);
    }

    #[test]
    fn a_tier_budget_never_drops_below_the_frame_minimum() {
        let processor = smart_processor();
        let clip = VideoClipOptions {
            sample_fps: 2.0,
            max_long_side_pixel: Some(28),
        };

        let layered = processor.for_clip(&clip);
        assert_eq!(layered.video_max_pixels(), layered.min_pixels());
    }

    #[test]
    fn video_tiers_change_the_token_count() {
        use crate::vision::processor::VisionPreProcessor;

        let processor = smart_processor();
        let tokens = |width: u32, height: u32, tier: Option<u32>| {
            let clip = VideoClipOptions {
                sample_fps: 2.0,
                max_long_side_pixel: tier,
            };
            let frames = vec![DynamicImage::new_rgb8(width, height); 4];
            processor
                .preprocess_video(&frames, &clip)
                .unwrap()
                .feature_token_counts[0]
        };

        // A 16:9 source, capped upstream to each tier's long side.
        let low = tokens(504, 284, Some(504));
        let mid = tokens(1008, 567, Some(1008));
        let high = tokens(2016, 1134, Some(2016));
        assert!(
            low < mid && mid < high,
            "tiers must order the token count: {low} < {mid} < {high}"
        );

        // Without a tier the reference per-frame budget (768 * 28 * 28) applies,
        // and a 2016-wide frame is shrunk onto the same grid as a 1008-wide one.
        assert_eq!(tokens(2016, 1134, None), tokens(1008, 567, None));
        assert_eq!(tokens(1008, 567, None), mid);
    }

    #[test]
    fn malformed_compression_values_are_rejected() {
        for bad in ["\"two\"", "0", "2.5", "null"] {
            let raw = format!(
                r#"{{"patch_size": 14,
                     "img_token_compression_config": {{"spatial_merge_size": {bad}}}}}"#
            );
            let config: PreProcessorConfig = serde_json::from_str(&raw).unwrap();
            let err = MiniMaxM3VisionProcessor::new().layered_over(&config);
            assert!(
                err.is_err(),
                "a present but malformed spatial_merge_size ({bad}) must fail loudly"
            );
        }
    }

    #[test]
    fn absent_compression_block_uses_defaults() {
        let config: PreProcessorConfig = serde_json::from_str(r#"{"patch_size": 14}"#).unwrap();
        let layered = MiniMaxM3VisionProcessor::new()
            .layered_over(&config)
            .unwrap();
        assert_eq!(layered.merge_size(), DEFAULT_MERGE_SIZE);
        assert_eq!(layered.temporal_patch_size(), DEFAULT_TEMPORAL_PATCH_SIZE);
    }

    #[test]
    fn temporal_patch_size_from_follows_the_layering_precedence() {
        // The flat key, then the checkpoint block, then the default.
        let mut config = m3_config();
        assert_eq!(
            MiniMaxM3VisionProcessor::temporal_patch_size_from(&config).unwrap(),
            2
        );
        config.temporal_patch_size = Some(4);
        assert_eq!(
            MiniMaxM3VisionProcessor::temporal_patch_size_from(&config).unwrap(),
            4
        );
        assert_eq!(
            MiniMaxM3VisionProcessor::temporal_patch_size_from(&PreProcessorConfig::default())
                .unwrap(),
            DEFAULT_TEMPORAL_PATCH_SIZE
        );
    }

    #[test]
    fn temporal_patch_size_from_rejects_unusable_block_values() {
        // A zero flat key defers to the block; a zero or malformed block value
        // is the configuration error the construction path reports, not the
        // default.
        let mut config = m3_config();
        config.temporal_patch_size = Some(0);
        assert_eq!(
            MiniMaxM3VisionProcessor::temporal_patch_size_from(&config).unwrap(),
            2
        );
        for bad in ["0", "-1", "2.5", "\"two\"", "null"] {
            let raw =
                format!(r#"{{"img_token_compression_config": {{"temporal_patch_size": {bad}}}}}"#);
            let config: PreProcessorConfig = serde_json::from_str(&raw).unwrap();
            let error = MiniMaxM3VisionProcessor::temporal_patch_size_from(&config)
                .unwrap_err()
                .to_string();
            assert!(error.contains("temporal_patch_size"), "{error}");
            let constructed = MiniMaxM3VisionProcessor::from_preprocessor_config(&config)
                .err()
                .unwrap()
                .to_string();
            assert_eq!(
                error, constructed,
                "temporal_patch_size {bad}: one reader, one error"
            );
        }
    }

    #[test]
    fn layering_pairs_frames_by_the_temporal_patch_size_the_lookup_reports() {
        let mut flat = m3_config();
        flat.temporal_patch_size = Some(4);
        let mut zero_flat = m3_config();
        zero_flat.temporal_patch_size = Some(0);
        let nested: PreProcessorConfig =
            serde_json::from_str(r#"{"img_token_compression_config": {"temporal_patch_size": 3}}"#)
                .unwrap();

        for config in [
            m3_config(),
            flat,
            zero_flat,
            nested,
            PreProcessorConfig::default(),
        ] {
            let layered = MiniMaxM3VisionProcessor::new()
                .layered_over(&config)
                .unwrap();
            assert_eq!(
                layered.temporal_patch_size(),
                MiniMaxM3VisionProcessor::temporal_patch_size_from(&config).unwrap()
            );
        }
    }

    #[test]
    fn from_preprocessor_config_rejects_a_malformed_compression_block() {
        // Construction must fail loudly rather than fall back to the defaults
        // and hide the real cause until the first request.
        let config: PreProcessorConfig = serde_json::from_str(
            r#"{"patch_size": 14,
                 "img_token_compression_config": {"spatial_merge_size": "two"}}"#,
        )
        .unwrap();
        assert!(MiniMaxM3VisionProcessor::from_preprocessor_config(&config).is_err());
    }

    #[test]
    fn video_preprocessing_is_supported() {
        use crate::vision::processor::VisionPreProcessor;

        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        // Two frames so the temporal patch pairing has something to pair.
        let frames = vec![
            DynamicImage::new_rgb8(224, 224),
            DynamicImage::new_rgb8(224, 224),
        ];

        // The shared Qwen base implements the video path; M3 must delegate to
        // it rather than falling through to the "unsupported" default.
        let out = processor
            .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
            .expect("M3 supports video preprocessing");
        assert!(!out.feature_token_counts.is_empty());
    }

    #[test]
    fn narrow_video_matches_minimax_reference() {
        use crate::vision::processor::ModelSpecificValue;

        // Generated by scripts/generate_minimax_narrow_video_golden.py using
        // the pinned official processor. Check every normalized patch value,
        // including channel/temporal ordering, as well as the grid and tokens.
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/golden/minimax_narrow_video.json"
        ))
        .unwrap();
        let processor = MiniMaxM3VisionProcessor::new();
        for case in fixture["cases"].as_array().unwrap() {
            let width = case["width"].as_u64().unwrap() as u32;
            let height = case["height"].as_u64().unwrap() as u32;
            let frames = fixture["colors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|color| {
                    DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                        width,
                        height,
                        image::Rgb(std::array::from_fn(|i| color[i].as_u64().unwrap() as u8)),
                    ))
                })
                .collect::<Vec<_>>();
            let output = processor
                .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
                .unwrap();
            let shape = case["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            assert_eq!(output.encoder_input.shape(), shape);
            assert_eq!(output.feature_token_counts, vec![shape[0] / 4]);
            let expected_grid = case["grid"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>();
            assert!(matches!(
                output.model_specific.get("video_grid_thw"),
                Some(ModelSpecificValue::IntTensor { data, shape })
                    if data == &expected_grid && shape == &[1, 3]
            ));
            for (index, &value) in output.encoder_input.as_f32().iter().enumerate() {
                let expected = case["channel_frame_values"][(index % 1176) / 196]
                    .as_f64()
                    .unwrap() as f32;
                assert!(
                    (value - expected).abs() < 1e-6,
                    "{width}x{height} patch value {index}: {value} != {expected}"
                );
            }
        }
    }

    #[test]
    fn video_zero_dimensions_are_rejected() {
        let processor = MiniMaxM3VisionProcessor::new();
        for (width, height) in [(0, 100), (100, 0), (0, 0)] {
            let frames = vec![DynamicImage::new_rgb8(width, height); 2];
            assert!(processor
                .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
                .is_err());
        }
    }

    #[test]
    fn short_sides_below_the_floor_are_raised_before_the_patch_grid() {
        let processor = smart_processor();
        // 400x40 is raised to 1120x112: an 8x80 patch grid, 160 tokens after the 2x2 merge.
        assert_eq!(processor.calculate_num_tokens(400, 40).unwrap(), 160);
        assert_eq!(processor.calculate_num_tokens(40, 400).unwrap(), 160);
        // A short side already at the floor is left alone.
        assert_eq!(processor.calculate_num_tokens(400, 112).unwrap(), 56);
    }

    #[test]
    fn raised_dimensions_raise_only_short_sides_below_the_floor() {
        assert_eq!(
            MiniMaxM3VisionProcessor::raised_dimensions(400, 40),
            Some((1120, 112))
        );
        assert_eq!(
            MiniMaxM3VisionProcessor::raised_dimensions(40, 400),
            Some((112, 1120))
        );
        assert_eq!(
            MiniMaxM3VisionProcessor::raised_dimensions(50, 50),
            Some((112, 112))
        );
        assert_eq!(
            MiniMaxM3VisionProcessor::raised_dimensions(112, 20),
            Some((627, 112))
        );
        assert_eq!(MiniMaxM3VisionProcessor::raised_dimensions(112, 400), None);
        assert_eq!(MiniMaxM3VisionProcessor::raised_dimensions(0, 40), None);
    }

    #[test]
    fn tokens_do_not_decrease_as_the_short_side_shrinks_within_the_pixel_budget() {
        // At a fixed long side, shrinking the short side below the floor raises
        // it by a larger factor, so the token count must not decrease while the
        // raised image still fits max_pixels (the verifier's 10_16 band).
        let processor = smart_processor();
        let tokens: Vec<usize> = [90, 40, 20]
            .into_iter()
            .map(|short| processor.calculate_num_tokens(400, short).unwrap())
            .collect();
        assert_eq!(tokens, vec![72, 160, 320]);
    }

    #[test]
    fn past_the_pixel_budget_the_uniform_clamp_wins() {
        // Beyond roughly 36:1 the raised image overshoots max_pixels, so the
        // base scales it down uniformly and the short side ends below 112
        // again, as the reference processor does; monotonicity is not kept.
        let processor = smart_processor();
        let tokens: Vec<usize> = [20, 10, 5]
            .into_iter()
            .map(|short| processor.calculate_num_tokens(400, short).unwrap())
            .collect();
        assert_eq!(tokens, vec![320, 453, 428]);
    }

    fn gradient(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y * 6 % 256) as u8, ((x + y) % 256) as u8])
        }))
    }

    #[test]
    fn the_raise_uses_the_requests_resample_filter() {
        // With `resample` = nearest, the wrapper's stage must produce the
        // pixels the base would get from a nearest-neighbour raise, and they
        // must differ from the bicubic ones.
        let image = gradient(400, 40);
        let mut nearest = smart_config();
        nearest.resampling = Some(0);
        let mut bicubic = smart_config();
        bicubic.resampling = Some(3);
        let nearest = MiniMaxM3VisionProcessor::from_preprocessor_config(&nearest).unwrap();
        let bicubic = MiniMaxM3VisionProcessor::from_preprocessor_config(&bicubic).unwrap();

        let via_wrapper = nearest.preprocess(std::slice::from_ref(&image)).unwrap();
        let staged = resize(&image, 1120, 112, FilterType::Nearest);
        let via_base = nearest.inner.preprocess(&[staged]).unwrap();
        assert_eq!(via_wrapper.encoder_input, via_base.encoder_input);

        let via_bicubic = bicubic.preprocess(&[image]).unwrap();
        assert_ne!(via_wrapper.encoder_input, via_bicubic.encoder_input);
    }

    #[test]
    fn do_resize_false_skips_the_raise() {
        // The caller asked for no resizing, so the floor does not apply and
        // the base rejects the off-grid buffer exactly as it did before.
        let mut config = smart_config();
        config.do_resize = Some(false);
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
        assert!(processor
            .preprocess(&[DynamicImage::new_rgb8(400, 40)])
            .is_err());
        assert_eq!(processor.calculate_num_tokens(400, 40).unwrap(), 14);
    }

    #[test]
    fn staged_images_stay_near_the_pixel_budget() {
        let processor = smart_processor();
        let cap = (processor.max_pixels() as f64 * 1.3) as usize;
        for (width, height) in [
            (200, 1),
            (150, 1),
            (1000, 6),
            (4000, 24),
            (400, 10),
            (400, 5),
        ] {
            let (w, h) = processor
                .staging_dimensions(width, height)
                .unwrap()
                .expect("below the floor");
            assert!(
                w as usize * h as usize <= cap,
                "{width}x{height} staged at {w}x{h}"
            );
            assert!(
                aspect_ratio(w as usize, h as usize) <= MAX_ASPECT_RATIO,
                "{width}x{height}"
            );
        }
        // Within the budget the raised image itself is staged.
        assert_eq!(
            processor.staging_dimensions(400, 40).unwrap(),
            Some((1120, 112))
        );
        // Over it, the base's own target is staged once its ratio is acceptable.
        assert_eq!(
            processor.staging_dimensions(400, 10).unwrap(),
            Some((4228, 84))
        );
    }

    #[test]
    fn preprocess_resizes_a_flat_image_onto_the_raised_grid() {
        let processor = smart_processor();
        let out = processor
            .preprocess(&[DynamicImage::new_rgb8(400, 40)])
            .expect("flat image preprocesses");
        assert_eq!(out.feature_token_counts, vec![160]);
        let crate::ModelSpecificValue::IntTensor { data, .. } =
            &out.model_specific["image_grid_thw"]
        else {
            panic!("image_grid_thw is an int tensor");
        };
        assert_eq!(data, &[1, 8, 80]);
    }

    #[test]
    fn token_count_matches_preprocess_for_raised_images() {
        // The placeholder count and the produced grid come from different
        // entry points; they must agree, including where the raised long side
        // rounds to the patch factor (112x20 -> 627x112 -> 616x112).
        let processor = smart_processor();
        for (width, height) in [(400, 40), (40, 400), (112, 20), (50, 50), (300, 80)] {
            let out = processor
                .preprocess(&[DynamicImage::new_rgb8(width, height)])
                .unwrap();
            assert_eq!(
                out.feature_token_counts,
                vec![processor.calculate_num_tokens(width, height).unwrap()],
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn thin_images_within_the_aspect_limit_still_preprocess() {
        // The base's 200:1 aspect guard must see the raised dimensions, whose
        // ratio equals the original's, never an intermediate grid.
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        for (width, height) in [(672, 4), (1008, 6), (4000, 24), (150, 1), (200, 1)] {
            let out = processor
                .preprocess(&[DynamicImage::new_rgb8(width, height)])
                .unwrap_or_else(|err| panic!("{width}x{height}: {err}"));
            assert_eq!(
                out.feature_token_counts,
                vec![processor.calculate_num_tokens(width, height).unwrap()],
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn off_contract_aspect_ratios_are_still_rejected_with_the_callers_dimensions() {
        // Raising keeps the aspect ratio, so the base's guard fires for the
        // same inputs as before and names the dimensions the caller sent.
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        for (width, height) in [(201, 1), (2, 500), (30_000, 100)] {
            let err = processor
                .preprocess(&[DynamicImage::new_rgb8(width, height)])
                .expect_err("aspect ratio above 200:1 is rejected");
            let TransformError::InvalidShape { actual, .. } = err else {
                panic!("{width}x{height}: {err}");
            };
            assert_eq!(actual, vec![height as usize, width as usize]);
        }
    }

    #[test]
    fn a_mixed_batch_with_an_off_contract_image_is_rejected_as_a_whole() {
        // One off-contract image rejects the batch before any pixel work.
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        let images = [
            DynamicImage::new_rgb8(112, 30_000),
            DynamicImage::new_rgb8(50, 50),
        ];
        let err = processor.preprocess(&images).expect_err("rejected");
        let TransformError::InvalidShape { actual, .. } = err else {
            panic!("{err}");
        };
        assert_eq!(actual, vec![30_000, 112]);
    }

    #[test]
    fn verifier_rule_b_cases_get_their_raised_grids() {
        // m3_image_tests 10_15: the five (width, height) cases and the token
        // count each produces once the short side is raised to 112.
        let processor = smart_processor();
        let tokens: Vec<usize> = [(400, 40), (300, 80), (112, 20), (40, 400), (80, 300)]
            .into_iter()
            .map(|(width, height)| processor.calculate_num_tokens(width, height).unwrap())
            .collect();
        assert_eq!(tokens, vec![160, 60, 88, 160, 60]);
    }

    #[test]
    fn token_count_matches_preprocess_under_pixel_overrides() {
        let mut config = smart_config();
        config.max_pixels = Some(100_352);
        config.min_pixels = Some(50_176);
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
        for (width, height) in [(400, 40), (50, 50), (672, 4), (300, 80), (400, 3)] {
            let out = processor
                .preprocess(&[DynamicImage::new_rgb8(width, height)])
                .unwrap();
            assert_eq!(
                out.feature_token_counts,
                vec![processor.calculate_num_tokens(width, height).unwrap()],
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn a_target_over_a_lowered_budget_is_not_staged() {
        // With max_pixels = 100,352 the base's target for 400x3 is 3640x28,
        // over the budget, so the base would scale it again; the shrunk raise
        // is staged instead and the grid is pinned.
        let processor = smart_processor();
        let mut config = smart_config();
        config.max_pixels = Some(100_352);
        let layered = processor.layered_over(&config).unwrap();
        assert_eq!(
            layered.staging_dimensions(400, 3).unwrap(),
            Some((4089, 31))
        );
        assert_eq!(layered.calculate_num_tokens(400, 3).unwrap(), 129);
    }

    #[test]
    fn thin_band_grids_are_pinned() {
        // Past 144:1 the raise is shrunk before staging; the grids are within
        // a percent of the full raise's (293 and 339 tokens).
        let processor = smart_processor();
        assert_eq!(processor.calculate_num_tokens(150, 1).unwrap(), 292);
        assert_eq!(processor.calculate_num_tokens(200, 1).unwrap(), 336);
    }

    #[test]
    fn mixed_batches_keep_their_order() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        let images = [
            DynamicImage::new_rgb8(224, 224),
            DynamicImage::new_rgb8(400, 40),
            DynamicImage::new_rgb8(224, 224),
        ];
        let out = processor.preprocess(&images).unwrap();
        // The aligned policy leaves 224x224 at 64 tokens and floors 400x40
        // up to 1680x168 (360 tokens).
        assert_eq!(out.feature_token_counts, vec![64, 360, 64]);
    }

    #[test]
    fn item_sizes_report_the_callers_dimensions() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        let images = [
            DynamicImage::new_rgb8(400, 40),
            DynamicImage::new_rgb8(224, 224),
        ];
        let out = processor.preprocess(&images).unwrap();
        assert_eq!(out.item_sizes, vec![(400, 40), (224, 224)]);
    }

    #[test]
    fn video_frames_follow_the_base_unchanged() {
        // Clips take the base's own path; under the aligned policy 400x40
        // frames floor up and cap to 672x84 (6x48 merged cells a pair).
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        let frames = vec![DynamicImage::new_rgb8(400, 40); 2];
        let out = processor
            .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
            .unwrap();
        let base = processor
            .inner
            .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
            .unwrap();
        assert_eq!(out.feature_token_counts, base.feature_token_counts);
        assert_eq!(out.feature_token_counts, vec![72]);
    }

    fn video_grid_thw(out: &PreprocessedEncoderInputs) -> Vec<i64> {
        let crate::ModelSpecificValue::IntTensor { data, .. } =
            &out.model_specific["video_grid_thw"]
        else {
            panic!("video_grid_thw is an int tensor");
        };
        data.clone()
    }

    #[test]
    fn the_video_budget_bounds_each_frame_not_the_sampled_volume() {
        // 720p lands on 560x1008 (720 tokens per pair) at any frame count; a volume budget gives 392x728 (364) for two frames.
        let processor = smart_processor();
        assert_eq!(processor.video_resize_mode(), QwenVideoResizeMode::PerFrame);
        assert_eq!(processor.video_max_pixels(), 602_112);
        assert_eq!(
            processor.smart_resize_video(2, 720, 1280).unwrap(),
            (560, 1008)
        );
        assert_eq!(
            processor.smart_resize_video(16, 720, 1280).unwrap(),
            (560, 1008)
        );
        let out = processor
            .preprocess_video(
                &vec![DynamicImage::new_rgb8(1280, 720); 2],
                &VideoClipOptions::sampled_at(2.0),
            )
            .unwrap();
        assert_eq!(video_grid_thw(&out), vec![1, 40, 72]);
        assert_eq!(out.feature_token_counts, vec![720]);

        let out = processor
            .preprocess_video(
                &vec![DynamicImage::new_rgb8(1280, 720); 16],
                &VideoClipOptions::sampled_at(2.0),
            )
            .unwrap();
        assert_eq!(video_grid_thw(&out), vec![8, 40, 72]);
        assert_eq!(out.feature_token_counts, vec![5760]);
    }

    #[test]
    fn large_images_are_bounded_by_max_pixels() {
        let processor = smart_processor();

        // Well past the bound: the count must clamp to image_seq_length.
        let tokens = processor.calculate_num_tokens(4096, 4096).unwrap();
        assert!(
            tokens <= 576,
            "expected at most image_seq_length (576) tokens, got {tokens}"
        );
    }

    #[test]
    fn smart_resize_floors_a_side_that_rounds_to_zero_like_the_reference() {
        // MiniMax-M3's reference takes `max(factor, round(side / factor) * factor)`
        // before the pixel-budget test, so a 1x100 image keeps its rounded long
        // side (28x112 = exactly min_pixels, no scale-up); the Qwen rule would
        // scale it up to 28x560.
        let processor = smart_processor();
        assert_eq!(processor.inner.smart_resize(1, 100).unwrap(), (28, 112));
        assert_eq!(processor.inner.smart_resize(100, 1).unwrap(), (112, 28));
    }

    // ====================================================================
    // Aligned long-side resize (the default policy)
    // ====================================================================

    /// The resized size `(width, height)` the aligned policy picks for an
    /// image, through the base's entry point.
    fn aligned_image_size(
        processor: &MiniMaxM3VisionProcessor,
        width: u32,
        height: u32,
    ) -> (u32, u32) {
        let (h, w) = processor
            .inner
            .smart_resize(height as usize, width as usize)
            .unwrap();
        (w as u32, h as u32)
    }

    #[test]
    fn aligned_resize_matches_the_m31_size_spec_examples() {
        // The MiniMax-M3.1 sizing spec's worked examples (image, detail auto):
        // width x height -> resized width x height = merged tokens.
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        for (width, height, want_w, want_h, tokens) in [
            (2016, 1134, 2016, 1148, 2952),
            (1920, 1080, 1932, 1092, 2691),
            (1024, 768, 1036, 784, 1036),
            (4000, 3000, 2016, 1540, 3960),
            (1000, 100, 1680, 168, 360),
        ] {
            assert_eq!(
                aligned_image_size(&processor, width, height),
                (want_w, want_h),
                "{width}x{height}"
            );
            assert_eq!(
                processor.calculate_num_tokens(width, height).unwrap(),
                tokens,
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn aligned_resize_detail_tiers_set_the_image_long_side_cap() {
        // A 4:3 image past every tier's cap, resized under each: low 672,
        // auto 2016 (the default), high 3584.
        let tokens = |detail: &str| {
            let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&aligned_config(
                serde_json::json!({"detail": detail}),
            ))
            .unwrap();
            processor.calculate_num_tokens(4000, 3000).unwrap()
        };
        assert_eq!(tokens("low"), 456); // 672x532
        assert_eq!(tokens("auto"), 3960); // 2016x1540
        assert_eq!(tokens("high"), 12416); // 3584x2716
    }

    #[test]
    fn aligned_resize_explicit_max_long_side_pixel_wins_and_clamps() {
        // An explicit cap wins over the detail tier...
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&aligned_config(
            serde_json::json!({"detail": "high", "max_long_side_pixel": 1008}),
        ))
        .unwrap();
        assert_eq!(processor.calculate_num_tokens(4000, 3000).unwrap(), 1008); // 1008x784

        // ...and is clamped to the high tier rather than rejected.
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&aligned_config(
            serde_json::json!({"max_long_side_pixel": 999_999}),
        ))
        .unwrap();
        assert_eq!(processor.calculate_num_tokens(4000, 3000).unwrap(), 12416); // 3584x2716

        // A cap below the alignment factor is a configuration error.
        assert!(
            MiniMaxM3VisionProcessor::from_preprocessor_config(&aligned_config(
                serde_json::json!({"max_long_side_pixel": 14}),
            ))
            .is_err()
        );
    }

    /// Under the aligned policy a request's image cap is the resize limit
    /// itself, as in the reference: 4000x3000 under 1008 aligns to 1008x784,
    /// where an image shrunk to 1008x756 first would stay 1008x756. The smart
    /// resize policy leaves the cap to the caller.
    #[test]
    fn a_request_image_cap_is_the_aligned_resize_limit() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        assert!(processor.applies_image_long_side());
        let out = processor
            .preprocess_with_context(
                &[DynamicImage::new_rgb8(4000, 3000)],
                &VisionPreprocessingContext {
                    image_long_side: Some(1008),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(out.feature_token_counts, vec![1008]); // 1008x784
        assert!(!smart_processor().applies_image_long_side());
    }

    #[test]
    fn aligned_resize_rejects_extreme_ratios_and_zero_dimensions() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        // 200:1 exactly is still accepted.
        assert!(processor.calculate_num_tokens(200, 1).is_ok());
        assert!(processor.calculate_num_tokens(201, 1).is_err());
        assert!(processor.calculate_num_tokens(1, 201).is_err());
        assert!(processor.calculate_num_tokens(0, 100).is_err());
        assert!(processor.calculate_num_tokens(100, 0).is_err());
    }

    #[test]
    fn aligned_resize_floors_tiny_images() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        // Below the 168^2 area floor and the 168 short-side floor.
        assert_eq!(aligned_image_size(&processor, 100, 100), (168, 168));
        assert_eq!(aligned_image_size(&processor, 50, 50), (168, 168));
        assert_eq!(processor.calculate_num_tokens(100, 100).unwrap(), 36);
        // The floors upscale past the old smart-resize minimum.
        assert_eq!(aligned_image_size(&processor, 400, 40), (1680, 168));
    }

    #[test]
    fn aligned_resize_ignores_the_legacy_pixel_bound_keys() {
        // min_pixels/max_pixels size the smart resize; under the aligned
        // policy they are legacy keys the M3.1 reference accepts and ignores.
        let mut config = m3_config();
        config.max_pixels = Some(3_136);
        config.min_pixels = Some(50_176);
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
        // 448x448 would clamp under the smart rule with a 3,136 area cap;
        // the aligned policy leaves it on its own grid.
        assert_eq!(processor.calculate_num_tokens(448, 448).unwrap(), 256);
    }

    #[test]
    fn aligned_resize_token_count_matches_preprocess() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        for (width, height) in [(400, 40), (100, 100), (4000, 3000), (112, 20), (2016, 1134)] {
            let out = processor
                .preprocess(&[DynamicImage::new_rgb8(width, height)])
                .unwrap();
            assert_eq!(
                out.feature_token_counts,
                vec![processor.calculate_num_tokens(width, height).unwrap()],
                "{width}x{height}"
            );
        }
    }

    #[test]
    fn aligned_video_caps_frames_at_the_detail_tier() {
        let grid_and_tokens = |detail: Option<&str>, clip_cap: Option<u32>| {
            let config = match detail {
                Some(detail) => aligned_config(serde_json::json!({"detail": detail})),
                None => m3_config(),
            };
            let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&config).unwrap();
            let frames = vec![DynamicImage::new_rgb8(1280, 720); 2];
            let clip = VideoClipOptions {
                sample_fps: 2.0,
                max_long_side_pixel: clip_cap,
            };
            let out = processor.preprocess_video(&frames, &clip).unwrap();
            (video_grid_thw(&out), out.feature_token_counts[0])
        };

        assert_eq!(grid_and_tokens(None, None), (vec![1, 28, 48], 336)); // 672x392
        assert_eq!(grid_and_tokens(Some("low"), None), (vec![1, 22, 36], 198)); // 504x308
        assert_eq!(grid_and_tokens(Some("high"), None), (vec![1, 52, 92], 1196)); // 1288x728
                                                                                  // A request's max_long_side_pixel wins over a non-auto detail.
        assert_eq!(
            grid_and_tokens(Some("high"), Some(1008)),
            (vec![1, 42, 72], 756) // 1008x588
        );
    }

    #[test]
    fn aligned_video_floors_upscale_small_frames() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        // 100x20 frames floor to 169x832 (the reference's float rounding) and
        // cap to the 672 long side: 672x168.
        let frames = vec![DynamicImage::new_rgb8(100, 20); 2];
        let out = processor
            .preprocess_video(&frames, &VideoClipOptions::sampled_at(2.0))
            .unwrap();
        assert_eq!(video_grid_thw(&out), vec![1, 12, 48]);
        assert_eq!(out.feature_token_counts, vec![144]);
    }

    #[test]
    fn aligned_video_rejects_clips_past_the_volume_cap() {
        let processor = MiniMaxM3VisionProcessor::from_preprocessor_config(&m3_config()).unwrap();
        // 720p frames resize to 392x672; the cap admits 1,142 of them.
        assert!(processor.smart_resize_video(1142, 720, 1280).is_ok());
        assert!(processor.smart_resize_video(1143, 720, 1280).is_err());
    }

    #[test]
    fn smart_resize_policy_reproduces_the_previous_sizing() {
        // The pre-aligned behavior stays selectable from the config.
        let processor = smart_processor();
        assert!(matches!(
            processor.resize_policy(),
            QwenResizePolicy::SmartResize
        ));
        assert_eq!(processor.calculate_num_tokens(448, 448).unwrap(), 256);
        assert!(processor.calculate_num_tokens(4096, 4096).unwrap() <= 576);
        // The 112 px short-side raise is part of the smart policy.
        assert_eq!(processor.calculate_num_tokens(400, 40).unwrap(), 160);
    }

    #[test]
    fn malformed_resize_policy_is_rejected_at_construction() {
        for bad in [
            serde_json::json!("aligned-long-side"),
            serde_json::json!({"kind": "aligned_long_side", "detail": "medium"}),
            serde_json::json!({"kind": "aligned_long_side", "surprise": 1}),
        ] {
            let config = aligned_config(bad.clone());
            assert!(
                MiniMaxM3VisionProcessor::from_preprocessor_config(&config).is_err(),
                "{bad}"
            );
        }
    }
}
