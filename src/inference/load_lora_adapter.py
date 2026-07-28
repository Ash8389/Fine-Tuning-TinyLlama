from peft import PeftModel

def load_finetuned_model(model, adapter):
    model = PeftModel.from_pretrained(
        model,
        adapter
    )

    return model