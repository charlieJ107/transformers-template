# 🤗 Transformers Template Project

A comprehensive template for training and evaluating deep learning models using PyTorch and the Hugging Face ecosystem. This template provides a well-structured foundation for NLP and multimodal projects with support for custom models, datasets, and training configurations.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Contributing](#contributing)
<!-- - [License](#license) -->

## ✨ Features

- **🏗️ Modular Architecture**: Clean separation of models, datasets, training, and utilities
- **🤗 Hugging Face Integration**: Built-in support for Transformers, Datasets, and Accelerate
- **⚡ Distributed Training**: Multi-GPU and multi-node training with Accelerate
- **📊 Comprehensive Logging**: Built-in experiment tracking and visualization
- **🔧 Flexible Configuration**: YAML-based configuration system
- **📦 Easy Deployment**: Support for both conda and pip environments
- **🧪 Testing Framework**: Structured testing and evaluation pipeline
- **📓 Jupyter Support**: Interactive development with notebook examples

## 🏗️ Project Structure

```
huggingface-template/
├── config/                     # Configuration files
│   ├── accelerate_config.yaml  # Accelerate configuration
│   └── training_args.yaml      # Training arguments
├── data/                       # Data directories
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   ├── interim/                # Intermediate data
│   └── external/               # External data sources
├── datasets/                   # Dataset implementations
│   └── example_dataset.py      # Example dataset class
├── models/                     # Model implementations
│   ├── pretrained_model/       # Custom pretrained models
│   │   ├── pretrained_model.py
│   │   └── pretrained_model_config.py
│   └── other/                  # Other model architectures
├── training/                   # Training scripts
│   ├── pretrained_model/       # Training scripts for pretrained models
│   │   └── train.py
│   └── other/                  # Other training scripts
├── processing/                 # Data processing utilities
│   └── my_processor.py         # Custom processor implementation
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── training_args.py        # Training argument utilities
├── visualization/              # Visualization utilities
│   └── visualization.py
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Documentation
├── main.py                     # Main entry point
├── environment.yml             # Conda environment
├── pyproject.toml             # Python project configuration
└── README.md                  # This file
```

## 🚀 Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/charlieJ107/huggingface-template.git
cd huggingface-template

# Install using uv (recommended)
uv sync
```

### Option 2: Using Conda

```bash
# Clone the repository
git clone https://github.com/charlieJ107/huggingface-template.git
cd huggingface-template

# Create and activate conda environment
conda env create -f environment.yml
conda activate my-project
```
### Devcontainer support

You may also use devcontainer to create your environment. Please check `.devcontainer` directory for details. 

## 🎯 Quick Start

### 1. Basic Usage

```bash
# Run the main script
python main.py
```

### 2. Training a Model

```bash
# Train with default configuration
python training/pretrained_model/train.py

# Train with custom configuration
python training/pretrained_model/train.py --config config/custom_training_args.yaml
```

### 3. Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ directory for examples
```

## ⚙️ Configuration

### Training Arguments

Edit `config/training_args.yaml` to customize training parameters:

```yaml
# Key training parameters
output_dir: "./outputs"
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
eval_strategy: "epoch"
save_strategy: "epoch"
logging_steps: 100
num_train_epochs: 3
learning_rate: 5e-5
warmup_steps: 500
```

### Accelerate Configuration

Configure distributed training in `config/accelerate_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 2
gpu_ids: [0, 1]
mixed_precision: fp16
```

You can also run accelerate CLI to make your configurations. 
```bash
accelerate config
```

## 📖 Usage

### Custom Models

1. **Create Model Configuration**:
   ```python
   # models/your_model/your_model_config.py
   from transformers import PretrainedConfig

   class YourModelConfig(PretrainedConfig):
       model_type = "your_model"
       
       def __init__(self, vocab_size=30522, **kwargs):
           super().__init__(**kwargs)
           self.vocab_size = vocab_size
   ```

2. **Implement Model**:
   ```python
   # models/your_model/your_model.py
   from transformers import PreTrainedModel
   from .your_model_config import YourModelConfig

   class YourModel(PreTrainedModel):
       config_class = YourModelConfig
       
       def __init__(self, config):
           super().__init__(config)
           # Your model implementation
   ```

### Custom Datasets

```python
# datasets/your_dataset.py
from torch.utils.data import Dataset

class YourDataset(Dataset):
    def __init__(self, data_path):
        # Load your data
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return processed sample
        return sample
```

### Training Scripts

```python
# training/your_model/train.py
from transformers import TrainingArguments, Trainer
from utils import load_training_args

# Load configuration
args = load_training_args("config/training_args.yaml")
training_args = TrainingArguments(**args)

# Initialize model, dataset, trainer
model = YourModel.from_pretrained("your-model-name")
train_dataset = YourDataset("data/train")
eval_dataset = YourDataset("data/eval")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start training
trainer.train()
```

## 🔧 Development

### Adding New Components

1. **Models**: Add to `models/` directory with config and implementation
2. **Datasets**: Add to `datasets/` directory with custom Dataset classes
3. **Training**: Add training scripts to `training/` directory
4. **Processing**: Add data processors to `processing/` directory

### Code Style

This project follows PEP 8 coding standards. Use tools like `black` and `flake8` for code formatting and linting.

### Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_models.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

<!-- ## 📝 License

This project is licensed under the MLP2.0 License - see the [LICENSE](LICENSE) file for details. -->

<!-- ## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/transformers-template/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/transformers-template/discussions)
- 📚 **Documentation**: [Project Wiki](https://github.com/yourusername/transformers-template/wiki) -->

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the amazing Transformers library
- [PyTorch](https://pytorch.org/) for the deep learning framework
- The open-source community for inspiration and contributions

---

**Happy coding! 🚀**