from trl import SFTTrainer

def load_trainer(model,  dataset, training_args):
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    return trainer