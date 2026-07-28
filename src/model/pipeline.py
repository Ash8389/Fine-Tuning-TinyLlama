from src.model.bits_bytes_config import bit_byte_config
from src.model.load_base_model import load_model
from src.model.lora_config import config_lora
from src.model.attach_lora import attach_lora

from peft import prepare_model_for_kbit_training



def pipeline(model_name):
    bnb_config = bit_byte_config()
    base_model = load_model(model_name=model_name, bnb_config=bnb_config)

    model = prepare_model_for_kbit_training(base_model)

    lora_config = config_lora()
    model = attach_lora(model=model, lora_config=lora_config)

    return base_model, model
    