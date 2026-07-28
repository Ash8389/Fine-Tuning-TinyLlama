from transformers import TrainingArguments


def training_argument():
    
    traingArgument = TrainingArguments(
        output_dir="outputs",

        overwrite_output_dir=True,

        num_train_epochs=1,

        per_device_train_batch_size=1,

        gradient_accumulation_steps=4,

        learning_rate=2e-4,

        logging_steps=10,

        save_strategy="epoch",

        fp16=True,

        report_to="none"
    )

    return traingArgument