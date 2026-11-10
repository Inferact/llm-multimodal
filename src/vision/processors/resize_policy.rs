//! Resize policies for the Qwen-style dynamic-resolution processors.
//!
//! A Qwen-style processor picks a resize target on the
//! `patch_size * merge_size` grid in one of two ways:
//!
//! - [`QwenResizePolicy::SmartResize`]: Qwen2-VL's `smart_resize`. Each side
//!   rounds to the nearest multiple of the factor and the total area is
//!   clamped to `[min_pixels, max_pixels]` (the bounds live in
//!   [`super::qwen_vl_base::QwenVLConfig`], alongside the per-video bounds).
//! - [`QwenResizePolicy::AlignedLongSide`]: MiniMax-M3.1's aligned long-side
//!   resize, mirroring `compute_aligned_size` of the public M3.1 processor
//!   (vLLM's vendored `vllm/transformers_utils/processors/minimax_m3.py`,
//!   itself mirroring sglang's). Tiny inputs are scaled up to an area floor
//!   and then to a short-side floor, both sides are rounded UP to the factor,
//!   and each side is capped at a per-modality long side. There is no area
//!   cap; the legacy `min_pixels`/`max_pixels` keys do not apply to it.
//!
//! # Configuration
//!
//! Processors that support both policies read the `resize_policy` key of the
//! preprocessor config (top level, landing in [`PreProcessorConfig::extra`]):
//!
//! - `"resize_policy": "smart_resize"` — the Qwen rule, sized by the flat
//!   `min_pixels`/`max_pixels` keys.
//! - `"resize_policy": "aligned_long_side"` — the aligned rule with the
//!   processor's built-in numbers.
//! - `"resize_policy": { "kind": "aligned_long_side", ... }` — the aligned
//!   rule with any of these fields overriding the built-in numbers:
//!   - `min_pixels` (default 28,224 = 168²): area floor.
//!   - `min_short_side` (default 168): short-side floor, applied after the
//!     area floor.
//!   - `max_ratio` (default 200): inputs with a larger aspect ratio are
//!     rejected.
//!   - `detail`: the OpenAI-style detail tier, `"low"` / `"auto"` / `"high"`,
//!     selecting the long-side cap for images and video (default `"auto"`).
//!   - `max_long_side_pixel`: explicit image long-side cap; wins over
//!     `detail`, is clamped to the `high` image cap, and must be at least the
//!     alignment factor.
//!   - `video_max_long_side_pixel`: explicit per-frame video long-side cap;
//!     wins over `detail` and must be at least the alignment factor. A
//!     request-level `max_long_side_pixel` on the clip wins over both.
//!   - `image_caps` / `video_caps`: `{"low": ..., "auto": ..., "high": ...}`
//!     long-side cap tables (images default to 672 / 2,016 / 3,584, video
//!     frames to 504 / 672 / 2,188).
//!   - `video_max_total_pixels` (default 301,056,000): hard volumetric cap;
//!     a clip whose `width * height * frames` exceeds it after resizing is
//!     rejected.
//!
//! Unknown fields, and values of the wrong type, are errors rather than
//! silently ignored: a mis-typed bound would otherwise change the token
//! geometry with no trace.

use crate::{
    types::ImageDetail,
    vision::{preprocessor_config::PreProcessorConfig, transforms::TransformError},
};

/// The preprocessor config key selecting (and tuning) the resize policy.
pub const RESIZE_POLICY_KEY: &str = "resize_policy";

/// The `low` / `auto` / `high` long-side caps of one modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetailLongSideCaps {
    pub low: u32,
    pub auto: u32,
    pub high: u32,
}

impl DetailLongSideCaps {
    /// The cap a `detail` tier selects.
    pub fn for_detail(&self, detail: ImageDetail) -> u32 {
        match detail {
            ImageDetail::Low => self.low,
            ImageDetail::Auto => self.auto,
            ImageDetail::High => self.high,
        }
    }
}

/// MiniMax-M3.1's aligned long-side resize: floor the area and the short
/// side, round both sides UP to the patch factor, cap each side at the
/// long-side cap (rounded down to the factor). No area cap.
///
/// The arithmetic mirrors the public reference (`compute_aligned_size` in
/// vLLM's vendored `minimax_m3.py`, mirroring sglang's M3.1 processor),
/// including its floating-point order: the floors scale by a double the
/// reference computes the same way, so `ceil(76 * (168 / 76))` is 169, not
/// 168, on both sides.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedLongSide {
    /// Area floor in pixels; smaller inputs are scaled up proportionally.
    pub min_pixels: usize,
    /// Short-side floor in pixels, applied after the area floor.
    pub min_short_side: u32,
    /// Inputs with a larger aspect ratio are rejected.
    pub max_ratio: u32,
    /// Per-tier long-side caps for images.
    pub image_caps: DetailLongSideCaps,
    /// Per-tier long-side caps for video frames.
    pub video_caps: DetailLongSideCaps,
    /// The detail tier selecting the caps (default auto).
    pub detail: ImageDetail,
    /// Explicit image long-side cap; wins over `detail`, clamped to
    /// `image_caps.high` at configuration time.
    pub image_max_long_side_pixel: Option<u32>,
    /// Explicit per-frame video long-side cap from the configuration; a
    /// request's own cap (or the per-clip override) wins over it.
    pub video_max_long_side_pixel: Option<u32>,
    /// Hard volumetric cap: `width * height * frames` after resizing.
    pub video_max_total_pixels: usize,
    /// The per-clip long-side cap a request set (see
    /// [`super::qwen_vl_base::QwenVLProcessorBase::with_video_long_side_override`]).
    pub(crate) video_long_side_override: Option<u32>,
}

impl AlignedLongSide {
    /// The long-side cap images are aligned under.
    pub fn image_long_side(&self) -> u32 {
        self.image_max_long_side_pixel
            .unwrap_or_else(|| self.image_caps.for_detail(self.detail))
    }

    /// The long-side cap video frames are aligned under: a request's
    /// per-clip cap, then the configuration's explicit cap, then the detail
    /// tier.
    pub fn video_long_side(&self) -> u32 {
        self.video_long_side_override
            .or(self.video_max_long_side_pixel)
            .unwrap_or_else(|| self.video_caps.for_detail(self.detail))
    }

    /// The aligned resize target `(height, width)` under `max_long_side`.
    ///
    /// Both results are multiples of `factor` and at most `max_long_side`
    /// rounded down to the factor. Zero dimensions, an aspect ratio past
    /// `max_ratio`, and a cap below the factor are rejected.
    pub fn aligned_size(
        &self,
        height: usize,
        width: usize,
        factor: usize,
        max_long_side: u32,
    ) -> Result<(usize, usize), TransformError> {
        if height == 0 || width == 0 {
            return Err(TransformError::InvalidShape {
                expected: "non-zero dimensions".to_string(),
                actual: vec![height, width],
            });
        }
        let max_long_side = max_long_side as usize;
        if max_long_side < factor {
            return Err(TransformError::ShapeError(format!(
                "aligned resize long-side cap {max_long_side} is below the alignment factor {factor}"
            )));
        }
        let ratio = height.max(width) as f64 / height.min(width) as f64;
        if ratio > f64::from(self.max_ratio) {
            return Err(TransformError::InvalidShape {
                expected: format!("aspect ratio < {}:1", self.max_ratio),
                actual: vec![height, width],
            });
        }

        // Floor layers, in the reference's floating-point arithmetic.
        let (mut height, mut width) = (height, width);
        if height * width < self.min_pixels {
            let beta = (self.min_pixels as f64 / (height * width) as f64).sqrt();
            height = (height as f64 * beta).ceil() as usize;
            width = (width as f64 * beta).ceil() as usize;
        }
        if height.min(width) < self.min_short_side as usize {
            let beta = f64::from(self.min_short_side) / height.min(width) as f64;
            height = (height as f64 * beta).ceil() as usize;
            width = (width as f64 * beta).ceil() as usize;
        }

        Ok(Self::align_to_factor(
            height,
            width,
            factor,
            max_long_side,
            max_long_side,
        ))
    }

    /// The aligned per-frame video target `(height, width)`, plus the hard
    /// volumetric cap: a clip whose resized `width * height * frames`
    /// exceeds `video_max_total_pixels` is rejected.
    pub fn aligned_video_size(
        &self,
        num_frames: usize,
        height: usize,
        width: usize,
        factor: usize,
    ) -> Result<(usize, usize), TransformError> {
        if num_frames == 0 {
            return Err(TransformError::InvalidShape {
                expected: "num_frames > 0".to_string(),
                actual: vec![num_frames],
            });
        }
        let (height, width) = self.aligned_size(height, width, factor, self.video_long_side())?;
        let volume = (width as u128) * (height as u128) * (num_frames as u128);
        if volume > self.video_max_total_pixels as u128 {
            return Err(TransformError::ShapeError(format!(
                "video area {volume} (width * height * frames) exceeds max_total_pixels {} after resizing",
                self.video_max_total_pixels
            )));
        }
        Ok((height, width))
    }

    /// Round both sides UP to `factor`, then cap each at its bound.
    ///
    /// The caps are rounded down to factor multiples first, so an aligned
    /// side never exceeds them. When either aligned side exceeds its cap,
    /// both scale back down proportionally in integer arithmetic and align
    /// again — the reference's `_align_to_factor`.
    fn align_to_factor(
        height: usize,
        width: usize,
        factor: usize,
        max_w: usize,
        max_h: usize,
    ) -> (usize, usize) {
        let max_w = max_w - max_w % factor;
        let max_h = max_h - max_h % factor;
        let align_up = |side: usize| {
            if side.is_multiple_of(factor) {
                side
            } else {
                side + (factor - side % factor)
            }
        };
        let mut new_w = align_up(width);
        let mut new_h = align_up(height);
        if new_w > max_w || new_h > max_h {
            let (aligned_w, aligned_h) = (new_w, new_h);
            new_w = (aligned_w * max_w / aligned_w).min(aligned_w * max_h / aligned_h);
            new_h = (aligned_h * max_w / aligned_w).min(aligned_h * max_h / aligned_h);
            new_w = align_up(new_w);
            new_h = align_up(new_h);
        }
        debug_assert!(
            new_w <= max_w && new_h <= max_h,
            "align_to_factor: (h={new_h}, w={new_w}) exceeds ({max_h}, {max_w})"
        );
        (new_h, new_w)
    }
}

/// How a Qwen-style processor picks each image's (or frame's) resize target.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum QwenResizePolicy {
    /// Qwen2-VL's `smart_resize`: nearest factor multiple per side, area
    /// clamped to `[min_pixels, max_pixels]` of
    /// [`super::qwen_vl_base::QwenVLConfig`].
    #[default]
    SmartResize,
    /// MiniMax-M3.1's aligned long-side resize; see [`AlignedLongSide`].
    AlignedLongSide(AlignedLongSide),
}

/// Read `key` of `map` as a `u64`; absent is `Ok(None)`, anything present
/// but not a non-negative integer is an error naming the key.
fn field_u64(
    map: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<u64>, TransformError> {
    match map.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(value) => match value.as_u64() {
            Some(parsed) => Ok(Some(parsed)),
            None => Err(TransformError::ShapeError(format!(
                "{RESIZE_POLICY_KEY}.{key} must be a non-negative integer, got {value}"
            ))),
        },
    }
}

/// Read a `{"low", "auto", "high"}` cap table, field by field over `base`.
fn field_caps(
    map: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    base: DetailLongSideCaps,
) -> Result<DetailLongSideCaps, TransformError> {
    let Some(value) = map.get(key) else {
        return Ok(base);
    };
    let Some(tiers) = value.as_object() else {
        return Err(TransformError::ShapeError(format!(
            "{RESIZE_POLICY_KEY}.{key} must be an object with low/auto/high, got {value}"
        )));
    };
    let mut caps = base;
    for (tier, slot) in [
        ("low", &mut caps.low),
        ("auto", &mut caps.auto),
        ("high", &mut caps.high),
    ] {
        if let Some(value) = field_u64(tiers, tier)? {
            *slot = u32::try_from(value).map_err(|_| {
                TransformError::ShapeError(format!(
                    "{RESIZE_POLICY_KEY}.{key}.{tier} does not fit in u32: {value}"
                ))
            })?;
        }
    }
    for unknown in tiers.keys() {
        if !["low", "auto", "high"].contains(&unknown.as_str()) {
            return Err(TransformError::ShapeError(format!(
                "{RESIZE_POLICY_KEY}.{key} has unknown tier {unknown:?}"
            )));
        }
    }
    Ok(caps)
}

/// Read a `detail` tier string.
fn field_detail(
    map: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<ImageDetail>, TransformError> {
    match map.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(detail)) => match detail.as_str() {
            "low" => Ok(Some(ImageDetail::Low)),
            "auto" => Ok(Some(ImageDetail::Auto)),
            "high" => Ok(Some(ImageDetail::High)),
            other => Err(TransformError::ShapeError(format!(
                "{RESIZE_POLICY_KEY}.{key} must be one of low/auto/high, got {other:?}"
            ))),
        },
        Some(value) => Err(TransformError::ShapeError(format!(
            "{RESIZE_POLICY_KEY}.{key} must be a string, got {value}"
        ))),
    }
}

impl QwenResizePolicy {
    /// The policy `config`'s `resize_policy` key selects, or `None` when the
    /// key is absent and the processor keeps what it was built with.
    ///
    /// `aligned_base` supplies the aligned numbers the object form layers
    /// its fields over: the processor's current aligned settings, or its
    /// family defaults when it is built on [`SmartResize`](Self::SmartResize).
    /// Caps below the alignment `factor` and a `max_ratio` of zero are
    /// rejected here, at construction.
    pub fn from_config(
        config: &PreProcessorConfig,
        aligned_base: &AlignedLongSide,
        factor: usize,
    ) -> Result<Option<QwenResizePolicy>, TransformError> {
        let Some(value) = config.extra.get(RESIZE_POLICY_KEY) else {
            return Ok(None);
        };
        if value.is_null() {
            return Ok(None);
        }
        if let Some(name) = value.as_str() {
            return match name {
                "smart_resize" => Ok(Some(Self::SmartResize)),
                "aligned_long_side" => {
                    Ok(Some(Self::AlignedLongSide(aligned_base.clone())))
                }
                other => Err(TransformError::ShapeError(format!(
                    "{RESIZE_POLICY_KEY} must be \"smart_resize\", \"aligned_long_side\" or an object, got {other:?}"
                ))),
            };
        }
        let Some(map) = value.as_object() else {
            return Err(TransformError::ShapeError(format!(
                "{RESIZE_POLICY_KEY} must be a string or an object, got {value}"
            )));
        };
        let kind = match map.get("kind") {
            None => "aligned_long_side",
            Some(serde_json::Value::String(kind)) => kind.as_str(),
            Some(other) => {
                return Err(TransformError::ShapeError(format!(
                    "{RESIZE_POLICY_KEY}.kind must be a string, got {other}"
                )));
            }
        };
        if kind == "smart_resize" {
            if map.len() > 1 {
                return Err(TransformError::ShapeError(format!(
                    "{RESIZE_POLICY_KEY} with kind \"smart_resize\" takes no other fields; \
                     the flat min_pixels/max_pixels keys size it"
                )));
            }
            return Ok(Some(Self::SmartResize));
        }
        if kind != "aligned_long_side" {
            return Err(TransformError::ShapeError(format!(
                "{RESIZE_POLICY_KEY}.kind must be \"smart_resize\" or \"aligned_long_side\", got {kind:?}"
            )));
        }

        let mut aligned = aligned_base.clone();
        if let Some(min_pixels) = field_u64(map, "min_pixels")? {
            aligned.min_pixels = min_pixels as usize;
        }
        if let Some(min_short_side) = field_u64(map, "min_short_side")? {
            aligned.min_short_side = u32::try_from(min_short_side).map_err(|_| {
                TransformError::ShapeError(format!(
                    "{RESIZE_POLICY_KEY}.min_short_side does not fit in u32: {min_short_side}"
                ))
            })?;
        }
        if let Some(max_ratio) = field_u64(map, "max_ratio")? {
            let max_ratio = u32::try_from(max_ratio).map_err(|_| {
                TransformError::ShapeError(format!(
                    "{RESIZE_POLICY_KEY}.max_ratio does not fit in u32: {max_ratio}"
                ))
            })?;
            if max_ratio == 0 {
                return Err(TransformError::ShapeError(format!(
                    "{RESIZE_POLICY_KEY}.max_ratio must be positive"
                )));
            }
            aligned.max_ratio = max_ratio;
        }
        if let Some(detail) = field_detail(map, "detail")? {
            aligned.detail = detail;
        }
        aligned.image_caps = field_caps(map, "image_caps", aligned.image_caps)?;
        aligned.video_caps = field_caps(map, "video_caps", aligned.video_caps)?;
        if let Some(cap) = field_u64(map, "max_long_side_pixel")? {
            aligned.image_max_long_side_pixel = Some(u32::try_from(cap).map_err(|_| {
                TransformError::ShapeError(format!(
                    "{RESIZE_POLICY_KEY}.max_long_side_pixel does not fit in u32: {cap}"
                ))
            })?);
        }
        if let Some(cap) = field_u64(map, "video_max_long_side_pixel")? {
            aligned.video_max_long_side_pixel = Some(u32::try_from(cap).map_err(|_| {
                TransformError::ShapeError(format!(
                    "{RESIZE_POLICY_KEY}.video_max_long_side_pixel does not fit in u32: {cap}"
                ))
            })?);
        }
        if let Some(volume) = field_u64(map, "video_max_total_pixels")? {
            aligned.video_max_total_pixels = volume as usize;
        }
        const KNOWN: [&str; 10] = [
            "kind",
            "min_pixels",
            "min_short_side",
            "max_ratio",
            "detail",
            "max_long_side_pixel",
            "video_max_long_side_pixel",
            "image_caps",
            "video_caps",
            "video_max_total_pixels",
        ];
        for unknown in map.keys() {
            if !KNOWN.contains(&unknown.as_str()) {
                return Err(TransformError::ShapeError(format!(
                    "{RESIZE_POLICY_KEY} has unknown field {unknown:?}"
                )));
            }
        }
        // An explicit image cap wins over the detail tier but is clamped to
        // the `high` cap, per the M3.1 size spec.
        if let Some(cap) = &mut aligned.image_max_long_side_pixel {
            *cap = (*cap).min(aligned.image_caps.high);
        }
        aligned.validate(factor)?;
        Ok(Some(Self::AlignedLongSide(aligned)))
    }
}

impl AlignedLongSide {
    /// Reject a configuration whose caps cannot hold even one aligned cell.
    fn validate(&self, factor: usize) -> Result<(), TransformError> {
        let factor = factor as u32;
        for (name, cap) in [
            ("image_caps.low", self.image_caps.low),
            ("image_caps.auto", self.image_caps.auto),
            ("image_caps.high", self.image_caps.high),
            ("video_caps.low", self.video_caps.low),
            ("video_caps.auto", self.video_caps.auto),
            ("video_caps.high", self.video_caps.high),
        ] {
            if cap < factor {
                return Err(TransformError::ShapeError(format!(
                    "{RESIZE_POLICY_KEY}.{name} ({cap}) is below the alignment factor {factor}"
                )));
            }
        }
        for (name, cap) in [
            ("max_long_side_pixel", self.image_max_long_side_pixel),
            ("video_max_long_side_pixel", self.video_max_long_side_pixel),
        ] {
            if let Some(cap) = cap {
                if cap < factor {
                    return Err(TransformError::ShapeError(format!(
                        "{RESIZE_POLICY_KEY}.{name} ({cap}) is below the alignment factor {factor}"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// The MiniMax-M3.1 numbers of the aligned policy: the public M3.1
/// processor's `compute_aligned_size` bounds and detail tables.
pub fn minimax_m31_aligned() -> AlignedLongSide {
    AlignedLongSide {
        min_pixels: 168 * 168,
        min_short_side: 168,
        max_ratio: 200,
        image_caps: DetailLongSideCaps {
            low: 672,
            auto: 2016,
            high: 3584,
        },
        video_caps: DetailLongSideCaps {
            low: 504,
            auto: 672,
            high: 2188,
        },
        detail: ImageDetail::Auto,
        image_max_long_side_pixel: None,
        video_max_long_side_pixel: None,
        video_max_total_pixels: 301_056_000,
        video_long_side_override: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> AlignedLongSide {
        minimax_m31_aligned()
    }

    #[test]
    fn worked_examples_of_the_m31_size_spec() {
        // The spec's image cases (detail auto, cap 2016), width x height ->
        // resized width x height and merged tokens.
        let policy = policy();
        for (width, height, want_w, want_h) in [
            (2016, 1134, 2016, 1148),
            (1920, 1080, 1932, 1092),
            (1024, 768, 1036, 784),
            (4000, 3000, 2016, 1540),
            (1000, 100, 1680, 168),
        ] {
            let (h, w) = policy.aligned_size(height, width, 28, 2016).unwrap();
            assert_eq!((w, h), (want_w, want_h), "{width}x{height}");
        }
    }

    #[test]
    fn floors_upscale_tiny_inputs() {
        let policy = policy();
        // Below the area floor: scaled to it, then aligned up.
        assert_eq!(policy.aligned_size(100, 100, 28, 2016).unwrap(), (168, 168));
        // The short-side floor scales the long side with it.
        assert_eq!(
            policy.aligned_size(100, 1000, 28, 2016).unwrap(),
            (168, 1680)
        );
        // The reference's float arithmetic: ceil(76 * (168 / 76)) is 169, so
        // 20x100 floors to 169x832 before the cap, not 168x832.
        assert_eq!(policy.aligned_size(20, 100, 28, 672).unwrap(), (168, 672));
    }

    #[test]
    fn the_cap_wins_over_the_floors() {
        let policy = policy();
        // 20x100 floors to 169x832, over the 672 video cap; the proportional
        // scale-down in integer math lands at 156x672 and aligns to 168x672.
        assert_eq!(policy.aligned_size(20, 100, 28, 672).unwrap(), (168, 672));
        assert_eq!(policy.aligned_size(100, 20, 28, 672).unwrap(), (672, 168));
    }

    #[test]
    fn rejects_zero_dimensions_and_extreme_ratios() {
        let policy = policy();
        assert!(policy.aligned_size(0, 100, 28, 2016).is_err());
        assert!(policy.aligned_size(100, 0, 28, 2016).is_err());
        assert!(policy.aligned_size(1, 201, 28, 2016).is_err());
        assert!(policy.aligned_size(1, 200, 28, 2016).is_ok());
    }

    #[test]
    fn rejects_a_cap_below_the_factor() {
        let policy = policy();
        assert!(policy.aligned_size(100, 100, 28, 14).is_err());
    }

    #[test]
    fn video_volume_cap_counts_the_resized_frames() {
        let policy = policy();
        // 720p frames resize to 392x672 under the video auto cap.
        let (h, w) = policy.aligned_video_size(1, 720, 1280, 28).unwrap();
        assert_eq!((h, w), (392, 672));
        let fits = 301_056_000usize / (392 * 672);
        assert_eq!(fits, 1142);
        assert!(policy.aligned_video_size(fits, 720, 1280, 28).is_ok());
        assert!(policy.aligned_video_size(fits + 1, 720, 1280, 28).is_err());
        assert!(policy.aligned_video_size(0, 720, 1280, 28).is_err());
    }

    #[test]
    fn detail_tiers_and_overrides_resolve_the_caps() {
        let mut policy = policy();
        assert_eq!(policy.image_long_side(), 2016);
        assert_eq!(policy.video_long_side(), 672);
        policy.detail = ImageDetail::Low;
        assert_eq!(policy.image_long_side(), 672);
        assert_eq!(policy.video_long_side(), 504);
        policy.detail = ImageDetail::High;
        assert_eq!(policy.image_long_side(), 3584);
        assert_eq!(policy.video_long_side(), 2188);
        // An explicit image cap wins over the detail tier.
        policy.image_max_long_side_pixel = Some(1008);
        assert_eq!(policy.image_long_side(), 1008);
        // The video order: per-clip override, then the config's explicit cap,
        // then the tier.
        policy.video_max_long_side_pixel = Some(1008);
        assert_eq!(policy.video_long_side(), 1008);
        policy.video_long_side_override = Some(504);
        assert_eq!(policy.video_long_side(), 504);
    }

    fn config_with_policy(value: serde_json::Value) -> PreProcessorConfig {
        let mut config = PreProcessorConfig::default();
        config.extra.insert(RESIZE_POLICY_KEY.to_string(), value);
        config
    }

    #[test]
    fn config_string_forms_select_the_policy() {
        let base = policy();
        assert_eq!(
            QwenResizePolicy::from_config(&config_with_policy("smart_resize".into()), &base, 28)
                .unwrap(),
            Some(QwenResizePolicy::SmartResize)
        );
        assert_eq!(
            QwenResizePolicy::from_config(
                &config_with_policy("aligned_long_side".into()),
                &base,
                28
            )
            .unwrap(),
            Some(QwenResizePolicy::AlignedLongSide(base.clone()))
        );
        assert_eq!(
            QwenResizePolicy::from_config(&PreProcessorConfig::default(), &base, 28).unwrap(),
            None
        );
        for bad in ["\"smart-resize\"", "0", "true", "[\"smart_resize\"]"] {
            let value = serde_json::from_str(bad).unwrap();
            assert!(
                QwenResizePolicy::from_config(&config_with_policy(value), &base, 28).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn config_object_layers_over_the_base() {
        let base = policy();
        let value = serde_json::json!({
            "kind": "aligned_long_side",
            "min_pixels": 40000,
            "detail": "high",
            "image_caps": {"high": 4000},
            "video_max_total_pixels": 1000
        });
        let Some(QwenResizePolicy::AlignedLongSide(aligned)) =
            QwenResizePolicy::from_config(&config_with_policy(value), &base, 28).unwrap()
        else {
            panic!("aligned policy");
        };
        assert_eq!(aligned.min_pixels, 40000);
        assert_eq!(aligned.detail, ImageDetail::High);
        assert_eq!(aligned.image_caps.high, 4000);
        assert_eq!(aligned.image_caps.auto, 2016, "untouched tiers survive");
        assert_eq!(aligned.video_max_total_pixels, 1000);
        assert_eq!(aligned.min_short_side, 168, "untouched fields survive");
    }

    #[test]
    fn config_object_rejects_unknown_and_malformed_fields() {
        let base = policy();
        for bad in [
            serde_json::json!({"kind": "aligned_long_side", "unknown": 1}),
            serde_json::json!({"kind": "aligned_long_side", "detail": "medium"}),
            serde_json::json!({"kind": "aligned_long_side", "min_pixels": "many"}),
            serde_json::json!({"kind": "aligned_long_side", "max_ratio": 0}),
            serde_json::json!({"kind": "aligned_long_side", "image_caps": {"huge": 4000}}),
            serde_json::json!({"kind": "smart_resize", "min_pixels": 3136}),
        ] {
            assert!(
                QwenResizePolicy::from_config(&config_with_policy(bad.clone()), &base, 28).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn explicit_image_cap_is_clamped_to_the_high_tier() {
        let base = policy();
        let value = serde_json::json!({"max_long_side_pixel": 999999});
        let Some(QwenResizePolicy::AlignedLongSide(aligned)) =
            QwenResizePolicy::from_config(&config_with_policy(value), &base, 28).unwrap()
        else {
            panic!("aligned policy");
        };
        assert_eq!(aligned.image_long_side(), 3584);
    }

    #[test]
    fn config_rejects_caps_below_the_factor() {
        let base = policy();
        for bad in [
            serde_json::json!({"image_caps": {"low": 14}}),
            serde_json::json!({"max_long_side_pixel": 14}),
            serde_json::json!({"video_max_long_side_pixel": 0}),
        ] {
            assert!(
                QwenResizePolicy::from_config(&config_with_policy(bad.clone()), &base, 28).is_err(),
                "{bad}"
            );
        }
    }
}
