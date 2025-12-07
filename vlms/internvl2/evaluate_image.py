# This file evaluates image QA performance on benchmark datasets.
import argparse
import json
from builtins import print, str
from pathlib import Path
from typing import Optional, Union

import datasets

from inference import load_image, load_internvl_model

RESULT_PATH = "results.jsonl"
DEFAULT_GENERATION_CONFIG = dict(max_new_tokens=1024, do_sample=True)


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


def load_dataset(dataset_path: Union[str, Path]):
    dataset_path = Path(dataset_path)
    print(f"Loading dataset from: {dataset_path}")
    dataset = datasets.load_from_disk(str(dataset_path))
    try:
        print(dataset)
    except Exception:
        print("dataset format: <unknown>")
    return dataset


def inference(model, tokenizer, image, query, generation_config=None, multichoice=False):
    """Run single-image inference using the InternVL chat API."""
    pixel_values = load_image(image, max_num=12)
    param = next(model.parameters())
    device = "cuda:1"#param.device
    dtype = param.dtype
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    config = generation_config or DEFAULT_GENERATION_CONFIG
    response = model.chat(tokenizer, pixel_values, query, generation_config=config)
    return response


def evaluate_dataset(
    dataset_path: Union[str, Path],
    image_key: str = "image",
    question_key: str = "question",
    multichoice_key: Optional[str] = None,
    split: str = "test",
    output_path: Union[str, Path] = RESULT_PATH,
    model_path: str = "OpenGVLab/InternVL2-8B",
    generation_config: Optional[dict] = None,
):
    """Run inference on every sample in the requested split and emit JSONL."""

    dataset = load_dataset(dataset_path)
    if split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split '{split}' not found. Available splits: {available}")

    model, tokenizer, _ = load_internvl_model(model_path)
    config = generation_config or DEFAULT_GENERATION_CONFIG
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as writer:
        for sample_idx, sample in enumerate(dataset[split]):
            if image_key not in sample:
                raise KeyError(f"Missing image key '{image_key}' in sample {sample_idx}")
            if question_key not in sample:
                raise KeyError(f"Missing question key '{question_key}' in sample {sample_idx}")

            image = sample[image_key]
            query = str(sample[question_key])
            choices = None
            if multichoice_key:
                if multichoice_key not in sample:
                    raise KeyError(
                        f"Missing multichoice key '{multichoice_key}' in sample {sample_idx}"
                    )
                choices = sample[multichoice_key]
                query_for_model = format_multichoice_prompt(query, choices)
            else:
                query_for_model = query
            response = inference(
                model,
                tokenizer,
                image,
                query_for_model,
                generation_config=config,
            )

            record = {
                "sample_index": sample_idx,
                "question": query,
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
    parser = argparse.ArgumentParser(description="Evaluate InternVL on an image QA dataset.")
    parser.add_argument("dataset_path", type=Path, help="Path to a datasets.load_from_disk folder")
    parser.add_argument("--output", type=Path, default=Path(RESULT_PATH), help="Where to save JSONL output")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    parser.add_argument("--image-key", default="image", help="Column name containing image data")
    parser.add_argument("--question-key", default="question", help="Column name containing prompts/questions")
    parser.add_argument(
        "--multichoice-key",
        default=None,
        help="Column name containing multiple-choice options to append",
    )
    parser.add_argument("--model-path", default="OpenGVLab/InternVL2-8B", help="Model identifier to load")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument(
        "--no-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling (defaults to enabled)",
    )
    parser.set_defaults(do_sample=True)
    return parser.parse_args()


def main():
    args = parse_args()
    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
    )
    evaluate_dataset(
        dataset_path=args.dataset_path,
        image_key=args.image_key,
        question_key=args.question_key,
        multichoice_key=args.multichoice_key,
        split=args.split,
        output_path=args.output,
        model_path=args.model_path,
        generation_config=generation_config,
    )


if __name__ == "__main__":
    main()