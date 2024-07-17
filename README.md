![Logo](./_fd23d3d2-3422-46a1-b15d-f6ecd4a65d4c.jpg)

# MyLLM

This repository contains a project that trains and generates text using a GPT-based language model enhanced with Retrieval-Augmented Generation (RAG). The model utilizes PyTorch for deep learning, Sentence-Transformers for encoding documents, and FAISS for efficient similarity search. There is the capability to pause training and resume at a later time, as well as early termination to avoid overfitting.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Text Generation](#text-generation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/gpt-language-model-rag.git
   cd gpt-language-model-rag
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, use the following command:

```bash
python MyLLM.py <path_to_text_file> [options]
```

**Arguments**:
- `filepath`: Path to the input text, PDF, or Parquet file.

**Options**:
- `--normalize`: Normalize text during preprocessing.
- `--use_subword`: Use subword tokenization.
- `--handle_unicode`: Handle Unicode characters during preprocessing.
- `--resume`: Resume training from the interim checkpoint.

**Example**:
```bash
python MyLLM.py data/sample.txt --normalize --use_subword --handle_unicode
```

### Text Generation

To generate text using the trained model, use the following command:

```bash
python MyLLM.py <path_to_text_file> --generate --start_text "<starting text>" [options]
```

**Options**:
- `--start_text`: Starting text for text generation.
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 100).
- `--temperature`: Sampling temperature for text generation (default: 1.0).

**Example**:
```bash
python MyLLM.py data/sample.txt --generate --start_text "Once upon a time" --max_new_tokens 50 --temperature 0.7
```

### Stopping and Restarting

The training process can be stopped and resumed using checkpoints. If the training is interrupted or the `--resume` flag is used, the model will continue training from the last saved checkpoint. Additionally, if the model reaches a specified target loss, it will stop training early, even if it hasn't completed the full number of epochs.

## Configuration

The configuration for the model is handled via the `ModelConfig` class within `MyLLM.py`. Key parameters include:

- `device`: Device to run the model on (`cuda` if available, else `cpu`).
- `batch_size`: Batch size for training.
- `block_size`: Size of input blocks.
- `max_iters`: Maximum number of training iterations.
- `learning_rate`: Learning rate for the optimizer.
- `eval_interval`: Interval for evaluation.
- `n_embd`: Dimension of the embeddings.
- `n_head`: Number of attention heads.
- `n_layer`: Number of transformer layers.
- `dropout`: Dropout rate.
- `num_workers`: Number of worker threads for data loading.
- `model_dir`, `log_dir`, `checkpoint_dir`: Directories for saving models, logs, and checkpoints.

## Project Structure

- `MyLLM.py`: Main script for training and text generation, contains the `ModelConfig` class.
- `data/`: Directory for storing input data.
- `logs/`: Directory for storing logs.
- `saved_models/`: Directory for storing saved models.
- `checkpoints/`: Directory for storing interim checkpoints.
- `requirements.txt`: List of required dependencies.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
