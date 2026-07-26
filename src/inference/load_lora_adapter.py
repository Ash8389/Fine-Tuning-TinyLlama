from peft import PeftModel

def load_finetuned_adapter(model, adapter):
    model = PeftModel.from_pretrained(
        model,
        adapter
    )

    return model