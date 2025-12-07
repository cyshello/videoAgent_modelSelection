import sys
from pathlib import Path
from typing import List

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

def build_messages(image_payload: Image.Image, query: str) -> List[dict]:
    """Construct chat messages for Qwen2-VL consumption."""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_payload},
                {"type": "text", "text": query},
            ],
        }
    ]


def run_inference(image_payload: Image.Image, query: str):
    messages = build_messages(image_payload, query)

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0] if output_text else "")


def frame_to_image(frame) -> Image.Image:
    """Convert an OpenCV BGR frame to a PIL RGB image."""

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python inference.py <video_path> <query> <frame_number>",
            file=sys.stderr,
        )
        sys.exit(1)

    video_path = Path(sys.argv[1]).expanduser()
    if not video_path.exists():
        print(f"Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    try:
        frame_number = int(sys.argv[-1])
    except ValueError:
        print("Frame number must be an integer.", file=sys.stderr)
        sys.exit(1)

    query = " ".join(sys.argv[2:-1]).strip()
    if not query:
        print("Query cannot be empty.", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        cap.release()
        print(
            f"Frame number {frame_number} is out of range. Total frames: {total_frames}",
            file=sys.stderr,
        )
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(
            f"Failed to read frame {frame_number} from video: {video_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        print("CUDA device not detected. Please run in a CUDA environment.", file=sys.stderr)
        sys.exit(1)

    image_payload = frame_to_image(frame)
    run_inference(image_payload, query)


if __name__ == "__main__":
    main()
