# Evaluate Qwen2-VL on an image QA dataset.
import argparse
import io
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import datasets
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

RESULT_PATH = "results.jsonl"
DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_DEVICE = "cuda"
DEFAULT_DEVICE_MAP = "auto"
DEFAULT_GENERATION_CONFIG = dict(
	max_new_tokens=256,
	temperature=0.0,
	top_p=1.0,
	repetition_penalty=1.0,
	do_sample=False,
)


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
	return f"{question}\nOptions:\n{options_block}"


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


def resolve_torch_dtype(name: Optional[str]):
	if not name:
		return "auto"
	normalized = name.strip().lower()
	mapping = {
		"float16": torch.float16,
		"fp16": torch.float16,
		"bfloat16": torch.bfloat16,
		"bf16": torch.bfloat16,
		"float32": torch.float32,
		"fp32": torch.float32,
	}
	if normalized == "auto":
		return "auto"
	if normalized not in mapping:
		raise ValueError(f"Unsupported torch dtype: {name}")
	return mapping[normalized]


def load_dataset(dataset_path: Union[str, Path]):
	dataset_path = Path(dataset_path)
	print(f"Loading dataset from: {dataset_path}")
	try:
		print("dataset format:", datasets.get_dataset_builder(str(dataset_path)))
	except Exception:
		print("dataset format: <unknown>")
	return datasets.load_from_disk(str(dataset_path))


def load_qwen_model(
	model_path: str = DEFAULT_MODEL_ID,
	device_map: str = DEFAULT_DEVICE_MAP,
	torch_dtype: Union[str, torch.dtype, None] = "auto",
):
	dtype = resolve_torch_dtype(torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
	if dtype == "auto":
		model = Qwen2VLForConditionalGeneration.from_pretrained(
			model_path,
			torch_dtype="auto",
			device_map=device_map,
		)
	else:
		model = Qwen2VLForConditionalGeneration.from_pretrained(
			model_path,
			torch_dtype=dtype,
			device_map=device_map,
		)
	processor = AutoProcessor.from_pretrained(model_path)
	model.eval()
	return model, processor


def build_messages(image_payload: Image.Image, query: str):
	return [
		{
			"role": "user",
			"content": [
				{"type": "image", "image": image_payload},
				{"type": "text", "text": query},
			],
		}
	]


def run_inference(
	image: Image.Image,
	query: str,
	model,
	processor,
	generation_config=None,
	device: str = DEFAULT_DEVICE,
):
	messages = build_messages(image, query)
	text = processor.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
	)
	image_inputs, video_inputs = process_vision_info(messages)
	inputs = processor(
		text=[text],
		images=image_inputs,
		videos=video_inputs,
		padding=True,
		return_tensors="pt",
	)
	inputs = inputs.to(device)

	config = DEFAULT_GENERATION_CONFIG.copy()
	if generation_config:
		config.update(generation_config)

	with torch.no_grad():
		generated_ids = model.generate(
			**inputs,
			max_new_tokens=config.get("max_new_tokens", 256),
			do_sample=config.get("do_sample", False),
			temperature=config.get("temperature", 0.0),
			top_p=config.get("top_p", 1.0),
			repetition_penalty=config.get("repetition_penalty", 1.0),
		)

	generated_ids_trimmed = [
		out_ids[len(in_ids) :]
		for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]
	output_text = processor.batch_decode(
		generated_ids_trimmed,
		skip_special_tokens=True,
		clean_up_tokenization_spaces=False,
	)
	return output_text[0] if output_text else ""


def evaluate_dataset(
	dataset_path: Union[str, Path],
	image_key: str = "image",
	question_key: str = "question",
	multichoice_key: Optional[str] = None,
	split: str = "test",
	output_path: Union[str, Path] = RESULT_PATH,
	model_path: str = DEFAULT_MODEL_ID,
	device: str = DEFAULT_DEVICE,
	device_map: str = DEFAULT_DEVICE_MAP,
	torch_dtype: Union[str, torch.dtype, None] = "auto",
	generation_config: Optional[dict] = None,
):
	dataset = load_dataset(dataset_path)
	if split not in dataset:
		available = ", ".join(dataset.keys())
		raise ValueError(f"Split '{split}' not found. Available splits: {available}")

	if device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA device not detected. Please run on a CUDA-enabled host.")

	model, processor = load_qwen_model(
		model_path=model_path,
		device_map=device_map,
		torch_dtype=torch_dtype,
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
				model=model,
				processor=processor,
				generation_config=generation_config,
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
	parser = argparse.ArgumentParser(description="Evaluate Qwen2-VL on an image QA dataset.")
	parser.add_argument("dataset_path", type=Path, help="Path to datasets.load_from_disk output")
	parser.add_argument("--output", type=Path, default=Path(RESULT_PATH), help="JSONL output path")
	parser.add_argument("--split", default="test", help="Dataset split to evaluate")
	parser.add_argument("--image-key", default="image", help="Column with images")
	parser.add_argument("--question-key", default="question", help="Column with questions")
	parser.add_argument(
		"--multichoice-key",
		default=None,
		help="Optional column containing multiple-choice options",
	)
	parser.add_argument("--model-path", default=DEFAULT_MODEL_ID, help="Model identifier")
	parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device for inputs (e.g., cuda)")
	parser.add_argument("--device-map", default=DEFAULT_DEVICE_MAP, help="Device map for model loading")
	parser.add_argument(
		"--torch-dtype",
		default="auto",
		help="Torch dtype (auto, float16, bfloat16, float32)",
	)
	parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
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
		device=args.device,
		device_map=args.device_map,
		torch_dtype=args.torch_dtype,
		generation_config=generation_config,
	)


if __name__ == "__main__":
	main()
