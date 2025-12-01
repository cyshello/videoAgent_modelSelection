import contextlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizer

model_name_or_path = "BAAI/EVA-CLIP-8B"  # or a local path to EVA-CLIP-8B weights
processor_name = "openai/clip-vit-large-patch14"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
amp_autocast = torch.cuda.amp.autocast if device.type == "cuda" else contextlib.nullcontext

processor = CLIPImageProcessor.from_pretrained(processor_name)
tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(
    model_name_or_path,
    torch_dtype=dtype,
    trust_remote_code=True,
).to(device)
model.eval()


def _encode_pil_image(image: Image.Image) -> np.ndarray:
    """Return a normalized EVA-CLIP embedding for a PIL image."""

    image_inputs = processor(images=image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device=device, dtype=dtype)

    with torch.no_grad():
        with amp_autocast():
            image_embedding = model.encode_image(pixel_values)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

    return image_embedding.squeeze(0).cpu().numpy().astype(np.float32)

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    return _encode_pil_image(image)

def get_video_embedding(
    video_path: str,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Generate 1 FPS embeddings for an .mp4 and persist them as <video_id>.npy.

    Args:
        video_path: Absolute or relative path to an .mp4 file.
        output_dir: Directory where the embedding .npy will be stored. Defaults to
            <repo>/embeddings next to this utils module when omitted.
        overwrite: Recompute embeddings even if the .npy already exists.

    Returns:
        Path to the saved .npy file containing shape (num_frames, embed_dim).
    """

    video_path = Path(video_path).expanduser().resolve()
    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"Only .mp4 inputs are supported, got {video_path.suffix}")

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "embeddings"
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{video_path.stem}.npy"
    if output_path.exists() and not overwrite:
        return output_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Processing video id:", video_path.stem)
    print("Video FPS:", fps)
    print("Video Frame Count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = fps if fps and fps > 0 else 30.0
    next_sample_time = 0.0
    second_stride = 1.0
    frame_idx = 0
    embeddings = []

    while True:
        success, frame = cap.read()
        if not success:
            break
        print(f"Processing frame {frame_idx}", end="\r")

        timestamp = frame_idx / fps
        if timestamp + 1e-6 >= next_sample_time:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embedding = _encode_pil_image(Image.fromarray(rgb))
            embeddings.append(embedding)
            next_sample_time += second_stride

        frame_idx += 1

    cap.release()

    if not embeddings:
        raise RuntimeError(f"No frames sampled from {video_path}")

    embedding_array = np.stack(embeddings)
    np.save(output_path, embedding_array)
    return output_path
    
if __name__ == "__main__":
    import sys
    folder_path = sys.argv[1]
    video_files = [f for f in Path(folder_path).iterdir() if f.suffix == ".mp4"]
    valid_path = "valid_videos.txt"
    valid = []
    with open(valid_path, "r") as f:
        for line in f:
            valid.append(line.strip())

    for video_file in video_files:
        if video_file.stem in valid:
            get_video_embedding(str(video_file),overwrite=True)