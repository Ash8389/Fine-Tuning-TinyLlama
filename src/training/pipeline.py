from src.training.training_argument import training_argument
from src.training.load_trainer import load_trainer

def pipeline(model, dataset):
    args = training_argument()
    trainer = load_trainer(model=model, dataset=dataset, training_args=args)

    trainer.train()

    return trainer
    