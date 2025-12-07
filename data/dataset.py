from datasets import load_dataset
from datasets import load_from_disk
import os

data_folder = "../imageqa"

def load_and_save_dataset(dataset_name, save_path):

    # Ensure the data_folder exists
    os.makedirs(data_folder, exist_ok=True)

    # Check if the dataset already exists in the drive
    if not os.path.exists(save_path):
        print(f"Downloading {dataset_name} dataset...")
        # Download the dataset
        if isinstance(dataset_name, str):
            dataset = load_dataset(dataset_name)
        else:
            dataset = load_dataset(*dataset_name)

        # Save the dataset locally (this will save it to the specified path within Google Drive)
        dataset.save_to_disk(save_path)
        print(f"Dataset saved to: {save_path}")
    else:
        loaded_dataset = load_from_disk(save_path)
        print(f"Dataset already exists at {save_path}. Skipping download.")
        print(loaded_dataset)

dataset_name = {"ChartQA" : "HuggingFaceM4/ChartQA",
                "ScienceQA" : ("lmms-lab/ScienceQA", "ScienceQA-IMG"),
                "DocVQA" : ("lmms-lab/DocVQA", "DocVQA"),
                "GQA_train_image" : ("lmms-lab/GQA", "train_balanced_images"),
                "GQA_train_instruction" : ("lmms-lab/GQA", "train_balanced_instructions")
                }


def main():
    for name,path in dataset_name.items():
        load_and_save_dataset(path, os.path.join(data_folder, name))

    ## Now, split GQA dataset into structure type.
    gqa = load_from_disk(os.path.join(data_folder, "GQA_train_instruction"))

    types = ["verify","query","choose","logical","compare"]
    gqa_splited = {}
    for t in types:
        if os.path.exists(os.path.join(data_folder,"GQA_"+t)):
            continue
        gqa_splited[t] = gqa.filter(lambda ex,t=t: ex['types']['structural'] == t)
        print(gqa_splited[t])
        gqa_splited[t].save_to_disk(os.path.join(data_folder,"GQA_"+t))

if __name__ == "__main__":
    main()