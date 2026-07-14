from peft import get_peft_model

def attach_lora(model, lora_config):
    model = get_peft_model(
        model,
        lora_config
    )

    return model