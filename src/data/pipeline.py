from src.data.data_loader import load_data
from src.data.to_chat_format import format_chat
from src.data.apply_template import chat_template
from src.data.filter_data import remove


def pipeline(dataset, tokenizer):
    dataset = load_data(dataset_name=dataset)
    
    chat_format_data = dataset.map(format_chat)

    templated_data = chat_format_data.map(
        lambda x: chat_template(x, tokenizer=tokenizer)
    )

    filtered_data = remove(templated_data)

    return filtered_data