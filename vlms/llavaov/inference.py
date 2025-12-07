# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from pathlib import Path

from PIL import Image
import cv2
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()
def frame_to_image(frame) -> Image.Image:
    """Convert an OpenCV BGR frame to a PIL RGB image."""

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def load_frame_from_video(video_path: Path, frame_number: int) -> Image.Image:
    """Grab a PIL image for the requested video frame."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        cap.release()
        raise ValueError(
            f"Frame number {frame_number} is out of range. Total frames: {total_frames}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(
            f"Failed to read frame {frame_number} from video: {video_path}"
        )

    return frame_to_image(frame)


def main():
    if len(sys.argv) < 4:
        script = sys.argv[0] if sys.argv else "inference.py"
        print(
            f"Usage: python {script} <video_path> <question> <frame_number>",
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

    user_question = " ".join(sys.argv[2:-1]).strip()
    if not user_question:
        print("Question cannot be empty.", file=sys.stderr)
        sys.exit(1)

    try:
        image = load_frame_from_video(video_path, frame_number)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + "\n" + user_question
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    image_sizes = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(text_outputs)


if __name__ == "__main__":
    main()
