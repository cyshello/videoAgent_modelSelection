from datasets import load_dataset
from datasets import load_from_disk
import os
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.utils.data import DataLoader
from datasets import Dataset


dataset_name = {"ChartQA" : "HuggingFaceM4/ChartQA",
                "ScienceQA" : ("lmms-lab/ScienceQA", "ScienceQA-IMG"),
                "DocVQA" : ("lmms-lab/DocVQA", "DocVQA"),
                "GQA_train_image" : ("lmms-lab/GQA", "train_balanced_images"),
                "GQA_train_instruction" : ("lmms-lab/GQA", "train_balanced_instructions")
                }

data_folder = "../../imageqa"

def load_and_save_dataset(dataset_name, save_path):

    # Ensure the data_folder exists
    os.makedirs(data_folder, exist_ok=True)

    # Check if the dataset already exists in the drive
    if not os.path.exists(save_path):
        print(f"Downloading {dataset_name} dataset...")
        # Download the dataset

        if isinstance(dataset_name, tuple):
            dataset = load_dataset(*dataset_name)
        else:
            dataset = load_dataset(dataset_name)

        # Save the dataset locally (this will save it to the specified path within Google Drive)
        dataset.save_to_disk(save_path)
        print(f"Dataset saved to: {save_path}")
    else:
        loaded_dataset = load_from_disk(save_path)
        print(f"Dataset already exists at {save_path}. Skipping download.")
        print(loaded_dataset)
        return loaded_dataset

dataset = {}
for name, path in dataset_name.items():
    dataset[name] = load_and_save_dataset(path, os.path.join(data_folder, name))

image_dict = {image['id']: image['image'] for image in dataset["GQA_train_image"]["train"]}

# 이미지와 쿼리 결합하는 함수
def extract_image_and_query(ex):
    image_id = ex["imageId"]
    image = image_dict.get(image_id, None)

    return {
        "image": image,
        "query": ex["question"],
        "label" : "internvl"
    }
transformed_dataset = dataset["GQA_train_instruction"]["train"][:10].map(extract_image_and_query)

dataset_field = {
    "ChartQA" : ["image", "query"],
    "ScienceQA" : ["image", "question"],
    "DocVQA" : ["question", "image"],
}

dataset_labels = {
    "ChartQA": "qwen2",
    "ScienceQA": "llavaov",
    "DocVQA": "qwen2"
}

for name, fields in dataset_field.items():
    def extract_image_and_query(ex):
        # 이미지와 쿼리 추출
        image = ex[fields[0]]  # 첫 번째 필드 (이미지)
        query = ex[fields[1]]  # 두 번째 필드 (쿼리)
        
        # 모델 라벨 추가
        model_name = dataset_labels[name]

        # 반환할 데이터
        return {
            "image": image,    # 이미지
            "query": query,    # 쿼리
            "label": model_name  # 모델 이름
        }

    # 'train' 데이터셋에서 이미지와 쿼리 결합
    dataset[name] = dataset[name]["train"][:10].map(extract_image_and_query)

all_data = []
for name in dataset_name.keys():
    all_data.extend(dataset[name])

all_data.extend(transformed_dataset)


final_dataset = Dataset.from_dict({
    "image": [data["image"] for data in all_data],
    "query": [data["query"] for data in all_data],
    "label": [data["label"] for data in all_data]
})

# DataLoader로 배치 처리
dataloader = DataLoader(final_dataset, batch_size=32, shuffle=True)

# DataLoader에서 첫 번째 배치 확인
for batch in dataloader:
    images, queries, labels = batch['image'], batch['query'], batch['label']
    print(images[0], len(queries), labels[0])  # 첫 번째 이미지, 쿼리 길이, 라벨 확인
    break 

