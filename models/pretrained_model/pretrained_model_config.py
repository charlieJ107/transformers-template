from transformers import PretrainedConfig


class MyPretrainedModelConfig(PretrainedConfig):

    # (str) — An identifier for the model type, serialized into the JSON file, and used to recreate the correct object in AutoConfig.
    model_type = "my_pretrained_model"
    # (bool) — Whether the config class can be initialized without providing input arguments. Some configurations requires inputs to be defined at init and have no default values, usually these are composite configs, (but not necessarily) such as EncoderDecoderConfig or ~RagConfig. They have to be initialized from two or more configs of type PretrainedConfig.
    has_no_defaults_at_init = True
    # (List[str]) — A list of keys to ignore by default when looking at dictionary outputs of the model during inference.
    keys_to_ignore_at_inference = []
    # (Dict[str, str]) — A dict that maps model specific attribute names to the standardized naming of attributes.
    attribute_map = {}
    # (Dict[str, Any]) — A dict that maps sub-modules FQNs of a base model to a tensor parallel plan applied to the sub-module when model.tensor_parallel is called.
    base_model_tp_plan = {}
    # (Dict[str, Tuple[List[str]]]) — A dict that maps child-modules of a base model to a pipeline parallel plan that enables users to place the child-module on the appropriate device.
    base_model_pp_plan = {}

    def __init__(
        self,
        vocab_size,
        hidden_size=2048,
        num_attention_heads=12,
        num_hidden_layers=6,
        **kwargs
    ):
        """A configuration class for a pretrained model.

        Args:
            vocab_size (int): The number of tokens in the vocabulary, which is also the first dimension of the embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
            hidden_size (int, optional): The hidden size of the model.. Defaults to 2048.
            num_attention_heads (int, optional):  The number of attention heads used in the multi-head attention layers of the model.. Defaults to 12.
            num_hidden_layers (int, optional): The number of blocks in the model. Defaults to 6.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

    @property
    def default_architectures(self):
        return ["MyPretrainedModel"]

    def validate(self):
        """Validates the configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError(
                f"Invalid vocab_size value: {self.vocab_size}. It should be a positive integer."
            )
