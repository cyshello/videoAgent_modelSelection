# Evaluate LLaVA-OneVision on an image QA dataset.
import argparse
import copy
import io
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import datasets
import numpy as np
import torch
from PIL import Image
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

RESULT_PATH = "results.jsonl"
DEFAULT_MODEL_ID = "lmms-lab/llava-onevision-qwen2-7b-ov"
DEFAULT_MODEL_NAME = "llava_qwen"
DEFAULT_DEVICE = "cuda"
DEFAULT_DEVICE_MAP = "auto"
DEFAULT_CONV_TEMPLATE = "qwen_1_5"
DEFAULT_GENERATION_CONFIG = dict(max_new_tokens=1024, do_sample=False, temperature=0.0)


def format_multichoice_prompt(question: str, choices) -> str:
	"""Append multiple-choice options to the question text."""

	if choices is None:
		return question

	def label_for_index(idx: int) -> str:
		base = ord("A") + idx
		if idx < 26:
			return chr(base)
		return str(idx + 1)

	if isinstance(choices, dict):
		option_lines = [f"{str(key)}. {value}" for key, value in choices.items()]
	elif isinstance(choices, (list, tuple)):
		option_lines = [
			f"{label_for_index(idx)}. {value}" for idx, value in enumerate(choices)
		]
	else:
		option_lines = [str(choices)]

	options_block = "\n".join(str(line) for line in option_lines)
	return f"Answer this multiple choice question based on the image, {question}\nOptions:\n{options_block} \n answer only with the number of the correct option."


def ensure_pil_image(image_source) -> Image.Image:
	"""Convert dataset image entries into RGB PIL images."""

	if isinstance(image_source, Image.Image):
		return image_source.convert("RGB")
	if isinstance(image_source, (str, Path)):
		return Image.open(image_source).convert("RGB")
	if isinstance(image_source, (bytes, bytearray)):
		return Image.open(io.BytesIO(image_source)).convert("RGB")
	if isinstance(image_source, np.ndarray):
		if image_source.dtype != np.uint8:
			image_source = image_source.astype(np.uint8)
		return Image.fromarray(image_source).convert("RGB")
	if isinstance(image_source, Sequence):
		array = np.array(image_source)
		if array.ndim >= 2:
			if array.dtype != np.uint8:
				array = array.astype(np.uint8)
			return Image.fromarray(array).convert("RGB")
	raise TypeError(f"Unsupported image type: {type(image_source)}")


def load_dataset(dataset_path: Union[str, Path]):
	dataset_path = Path(dataset_path)
	print(f"Loading dataset from: {dataset_path}")
	try:
		print("dataset format:", datasets.get_dataset_builder(str(dataset_path)))
	except Exception:
		print("dataset format: <unknown>")
	return datasets.load_from_disk(str(dataset_path))


def load_llava_model(
	model_path: str = DEFAULT_MODEL_ID,
	model_name: str = DEFAULT_MODEL_NAME,
	device: str = DEFAULT_DEVICE,
	device_map: str = DEFAULT_DEVICE_MAP,
):
	tokenizer, model, image_processor, _ = load_pretrained_model(
		model_path,
		None,
		model_name,
		device_map=device_map,
	)
	model.eval()
	return tokenizer, model, image_processor, device


def run_inference(
	image: Image.Image,
	query: str,
	tokenizer,
	model,
	image_processor,
	generation_config=None,
	conv_template: str = DEFAULT_CONV_TEMPLATE,
	device: str = DEFAULT_DEVICE,
):
	"""Run single-image inference using LLaVA-OneVision."""

	image_tensor = process_images([image], image_processor, model.config)
	image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

	conv = copy.deepcopy(conv_templates[conv_template])
	question = DEFAULT_IMAGE_TOKEN + "\n" + query
	conv.append_message(conv.roles[0], question)
	conv.append_message(conv.roles[1], None)
	prompt = conv.get_prompt()

	input_ids = tokenizer_image_token(
		prompt,
		tokenizer,
		IMAGE_TOKEN_INDEX,
		return_tensors="pt",
	).unsqueeze(0).to(device)
	image_sizes = [image.size]

	config = DEFAULT_GENERATION_CONFIG.copy()
	if generation_config:
		config.update(generation_config)

	generation_kwargs = dict(
		do_sample=config.get("do_sample", False),
		temperature=config.get("temperature", 0.0),
		top_p=config.get("top_p", 1.0),
		max_new_tokens=config.get("max_new_tokens", 1024),
		repetition_penalty=config.get("repetition_penalty", 1.0),
	)

	with torch.inference_mode():
		outputs = model.generate(
			input_ids,
			images=image_tensor,
			image_sizes=image_sizes,
			**generation_kwargs,
		)
	text_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
	return text_outputs[0] if text_outputs else ""


def evaluate_dataset(
	dataset_path: Union[str, Path],
	image_key: str = "image",
	question_key: str = "question",
	multichoice_key: Optional[str] = None,
	split: str = "test",
	output_path: Union[str, Path] = RESULT_PATH,
	model_path: str = DEFAULT_MODEL_ID,
	model_name: str = DEFAULT_MODEL_NAME,
	device: str = DEFAULT_DEVICE,
	device_map: str = DEFAULT_DEVICE_MAP,
	conv_template: str = DEFAULT_CONV_TEMPLATE,
	generation_config: Optional[dict] = None,
):
	"""Iterate over a dataset split, run inference, and write JSONL results."""

	dataset = load_dataset(dataset_path)
	if split not in dataset:
		available = ", ".join(dataset.keys())
		raise ValueError(f"Split '{split}' not found. Available splits: {available}")

	tokenizer, model, image_processor, device = load_llava_model(
		model_path=model_path,
		model_name=model_name,
		device=device,
		device_map=device_map,
	)

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", encoding="utf-8") as writer:
		for sample_idx, sample in enumerate(dataset[split]):
			if image_key not in sample:
				raise KeyError(f"Missing image key '{image_key}' in sample {sample_idx}")
			if question_key not in sample:
				raise KeyError(f"Missing question key '{question_key}' in sample {sample_idx}")

			image = ensure_pil_image(sample[image_key])
			question = str(sample[question_key])
			choices = None
			query = question
			if multichoice_key:
				if multichoice_key not in sample:
					raise KeyError(
						f"Missing multichoice key '{multichoice_key}' in sample {sample_idx}"
					)
				choices = sample[multichoice_key]
				query = format_multichoice_prompt(question, choices)

			response = run_inference(
				image=image,
				query=query,
				tokenizer=tokenizer,
				model=model,
				image_processor=image_processor,
				generation_config=generation_config,
				conv_template=conv_template,
				device=device,
			)

			record = {
				"sample_index": sample_idx,
				"question": question,
				"model_response": response,
			}
			if choices is not None:
				record["choices"] = choices
			if "id" in sample:
				record["sample_id"] = sample["id"]
			if "answer" in sample:
				record["ground_truth"] = sample["answer"]

			writer.write(json.dumps(record, ensure_ascii=False) + "\n")

	return output_path


def parse_args():
	parser = argparse.ArgumentParser(
		description="Evaluate LLaVA-OneVision on an image QA dataset."
	)
	parser.add_argument("dataset_path", type=Path, help="Path to datasets.load_from_disk output")
	parser.add_argument("--output", type=Path, default=Path(RESULT_PATH), help="JSONL output path")
	parser.add_argument("--split", default="test", help="Dataset split to evaluate")
	parser.add_argument("--image-key", default="image", help="Column that holds images")
	parser.add_argument("--question-key", default="question", help="Column with questions/prompts")
	parser.add_argument(
		"--multichoice-key",
		default=None,
		help="Optional column containing multiple-choice options",
	)
	parser.add_argument("--model-path", default=DEFAULT_MODEL_ID, help="Model identifier")
	parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name for loader")
	parser.add_argument("--device", default=DEFAULT_DEVICE, help="Compute device (e.g., cuda, cpu)")
	parser.add_argument("--device-map", default=DEFAULT_DEVICE_MAP, help="Device map passed to loader")
	parser.add_argument(
		"--conv-template",
		default=DEFAULT_CONV_TEMPLATE,
		help="Conversation template key for llava",
	)
	parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max generation length")
	parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
	parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top-p")
	parser.add_argument(
		"--repetition-penalty", type=float, default=1.0, help="Repetition penalty"
	)
	parser.add_argument(
		"--do-sample",
		action="store_true",
		help="Enable sampling (defaults to greedy)",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	generation_config = dict(
		max_new_tokens=args.max_new_tokens,
		temperature=args.temperature,
		top_p=args.top_p,
		repetition_penalty=args.repetition_penalty,
		do_sample=args.do_sample,
	)
	evaluate_dataset(
		dataset_path=args.dataset_path,
		image_key=args.image_key,
		question_key=args.question_key,
		multichoice_key=args.multichoice_key,
		split=args.split,
		output_path=args.output,
		model_path=args.model_path,
		model_name=args.model_name,
		device=args.device,
		device_map=args.device_map,
		conv_template=args.conv_template,
		generation_config=generation_config,
	)


if __name__ == "__main__":
	main()
