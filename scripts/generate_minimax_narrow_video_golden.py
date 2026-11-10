"""Generate narrow-frame expectations with the public MiniMax M3.1 video processor.

The reference is vLLM's vendored MiniMax M3.1 processor (Apache-2.0), pinned by
commit and verified by digest; its `compute_aligned_size` mirrors sglang's M3.1
processor. Requires transformers==5.20.2, torch and torchvision.
Run from the repository root.
"""

import hashlib
import importlib.util
import json
import tempfile
import urllib.request
from pathlib import Path

import torch

REVISION = "9b9bd049a73617e996d06cf35e6baf2a29ed03ae"
SOURCE = f"https://raw.githubusercontent.com/vllm-project/vllm/{REVISION}/vllm/transformers_utils/processors/minimax_m3.py"
SOURCE_SHA256 = "0cd5c6ac89a915c66766a0d4f5facb1adc61856e989d9bf84126e2506fd26054"
COLORS = [[32, 96, 224], [224, 64, 16]]


def main():
    payload = urllib.request.urlopen(SOURCE).read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            f"MiniMax-M3.1 reference at {SOURCE} has sha256 {digest}, "
            f"expected {SOURCE_SHA256}; pin a new revision deliberately"
        )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "minimax_m3_reference.py"
        path.write_bytes(payload)
        spec = importlib.util.spec_from_file_location("minimax_m3_reference", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        processor = module.MiniMaxM3VLVideoProcessor()
        cases = []
        for width, height in [(100, 20), (20, 100), (1, 1), (28, 28), (100, 100)]:
            video = torch.tensor(COLORS, dtype=torch.uint8)[:, :, None, None]
            video = video.expand(2, 3, height, width).contiguous()
            # Call the processor directly: frames are already sampled. The
            # aligned long-side resize (detail "auto": 672 px cap) applies.
            output = processor(videos=[video], do_resize=True, return_tensors="pt")
            pixels = output["pixel_values_videos"]
            # Each frame has constant RGB channels. Verify that every spatial
            # patch is identical before storing one value per channel/frame.
            blocks = pixels.reshape(-1, 3, 2, 14 * 14)
            values = blocks[0, :, :, 0]
            torch.testing.assert_close(blocks, values[None, :, :, None].expand_as(blocks))
            cases.append(
                {
                    "width": width,
                    "height": height,
                    "grid": output["video_grid_thw"][0].tolist(),
                    "shape": list(pixels.shape),
                    "channel_frame_values": values.flatten().tolist(),
                }
            )
        destination = Path("crates/multimodal/tests/fixtures/golden/minimax_narrow_video.json")
        destination.write_text(
            json.dumps(
                {
                    "source": SOURCE,
                    "source_sha256": SOURCE_SHA256,
                    "colors": COLORS,
                    "cases": cases,
                },
                indent=2,
            )
            + "\n"
        )
        print(destination.read_text())


if __name__ == "__main__":
    main()
