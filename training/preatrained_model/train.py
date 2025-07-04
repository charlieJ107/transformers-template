from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from models.pretrained_model.pretrained_model import MyPretrainedModel
from models.pretrained_model.pretrained_model_config import MyPretrainedModelConfig
from datasets.example_dataset import ExampleDataset
from processing.my_processor import MyProcessor
from utils import load_training_args


# Load training arguments from a YAML file
training_args = TrainingArguments(**load_training_args("config/training_args.yaml"))

model_config = MyPretrainedModelConfig(
    vocab_size=30522,  # Example vocab size, adjust as needed
)
model = MyPretrainedModel(config=model_config)

train_dataset = load_dataset("imdb", split="train")
eval_dataset = ExampleDataset(data=[{"text": "example text", "label": 0}])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=MyProcessor(tokenizer_name_or_path="bert-base-uncased"),
    compute_loss_func=None,  # Define your custom loss function if needed
    compute_metrics=None,  # Define your custom metrics function if needed
    callbacks=None,  # Define your custom callbacks if needed, default to all
    optimizers=(None, None),  # Define your custom optimizers if needed
    # Define your custom optimizer class and kwargs if needed
    optimizer_cls_and_kwargs=None,
    # Define your custom preprocessing function for logits if needed
    preprocess_logits_for_metrics=None,
)

trainer.train()
