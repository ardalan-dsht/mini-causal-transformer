# Mini Generative Pre-Trained Transformer
This repository contains a Python module for a MiniGPT (Mini Generative Pre-trained Transformer) model. The module is designed for language modeling and text generation tasks. It also includes a usage example in the form of a Jupyter Notebook (usage.ipynb) and a configuration file (config.py) to store model configurations.
### Educational Purpose
This repository is intended for educational purposes only.   
It serves as a learning resource to understand the inner workings of transformers and generative models.
Files

- model.py: Python module containing the MiniGPT model implementation.
- config.py: Configuration file storing model parameters.
- usage.ipynb: Jupyter Notebook demonstrating how to use the MiniGPT model.

Usage
To use the MiniGPT model, follow the instructions in the usage.ipynb notebook. The config.py file can be modified to adjust the model configurations according to specific requirements.
Example

```python
# Example usage of the MiniGPT model
import torch
from model import MiniGPT

# Instantiate the MiniGPT model
model = MiniGPT(config)

# Perform text generation
generated_text = model.generate(input_sequence, max_length)
```

Dependencies
The code in this repository has the following dependencies:

    PyTorch
