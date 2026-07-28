from src.inference.load_lora_adapter import load_finetuned_model
from src.inference.generate_basemodel_response import generate_respose

from transformers import AutoTokenizer

def pipeline(base_model, prompt):

    adapter_path = "outputs/final_adapter"

    finetuned_model = load_finetuned_model(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    basemodel_response = generate_respose(base_model, tokenizer=tokenizer, prompt=prompt)
    finetuned_response = generate_respose(finetuned_model, tokenizer=tokenizer, prompt=prompt)

    print(f"Fine tuned model res:\n {finetuned_response}")
    print(f"Base model res:\n {basemodel_response}")