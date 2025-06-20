from transformers import PreTrainedModel
import torch.nn as nn
from .pretrained_model_config import MyPretrainedModelConfig
from transformers import PretrainedConfig


def load_tf_weights(model: PreTrainedModel, config: PretrainedConfig, path: str = None):
    """
    Load TensorFlow weights into a PyTorch model.

    Args:
        model (PreTrainedModel): The PyTorch model to load weights into.
        config (PretrainedConfig): The configuration for the model.
        path (str, optional): The path to the TensorFlow checkpoint file. If None, it will use the default path.

    Returns:
        PreTrainedModel: The model with loaded weights.
    """
    # This function is a placeholder and should be implemented based on the specific model architecture.
    raise NotImplementedError("TensorFlow weight loading is not implemented.")


class MyPretrainedModel(PreTrainedModel):
    
    ## Class attributes
    # (PretrainedConfig) — A subclass of PretrainedConfig to use as configuration class for this model architecture.
    config_class = MyPretrainedModelConfig
    # (Callable) — A python method for loading a TensorFlow checkpoint in a PyTorch model, taking as arguments:
    load_tf_weights = load_tf_weights
    # (str) — A string indicating the attribute associated to the base model in derived classes of the same architecture adding modules on top of the base model.
    base_model_prefix = "my_pretrained"
    # (bool) — A flag indicating whether this model supports model parallelization.
    is_parallelizable = False
    # (str) — The name of the principal input to the model (often input_ids for NLP models, pixel_values for vision models and input_values for speech models).
    main_input_name = "input_ids"

    def __init__(self, config: MyPretrainedModelConfig):
        super().__init__(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits
