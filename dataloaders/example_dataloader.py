from torch.utils.data import Dataset, DataLoader

class ExampleDataset(Dataset):
    """
    Example dataset class for demonstration purposes.
    This class should be extended and customized based on the specific dataset requirements.
    """

    def __init__(self, data):
        """
        Initialize the dataset with data.

        Args:
            data (list): A list of data samples.
        """
        self.data = data

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Any: The sample at the specified index.
        """
        return self.data[idx]
    

example_data_loader = DataLoader(
    ExampleDataset(data=[{"text": "example text", "label": 0}]),
    batch_size=2,
    shuffle=True,
    num_workers=0  # Adjust based on your system's capabilities
)