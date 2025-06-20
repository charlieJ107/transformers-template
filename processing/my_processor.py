from transformers import AutoTokenizer, ProcessorMixin

class MyProcessor(ProcessorMixin):
    """
    A processor for the MyPretrainedModel that handles tokenization and other preprocessing tasks.
    """

    def __init__(self, tokenizer_name_or_path: str):
        """
        Initializes the processor with a tokenizer.

        Args:
            tokenizer_name_or_path (str): The name or path of the tokenizer to use.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def __call__(self, texts, **kwargs):
        """
        Processes the input texts by tokenizing them.

        Args:
            texts (list of str): The input texts to process.

        Returns:
            dict: A dictionary containing the tokenized inputs.
        """
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", **kwargs)