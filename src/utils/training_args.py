import yaml


def load_training_args(args_file_path: str) -> dict:
    """
    Load training arguments from a YAML file.

    Args:
        args_file_path (str): Path to the YAML file containing training arguments.

    Returns:
        dict: A dictionary containing the training arguments.
    """
    with open(args_file_path, 'r') as file:
        args = yaml.safe_load(file)

    return args
