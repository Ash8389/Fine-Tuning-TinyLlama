from datasets import load_dataset

def load_data(dataset_name):
    dataset = load_dataset(
        dataset_name,
        split="train"
    )

    return dataset