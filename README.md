![MyLLM](_fd23d3d2-3422-46a1-b15d-f6ecd4a65d4c.jpg)
## MyLLM: A Customizable Language Model Trainer

### Overview

MyLLM is a Python-based tool for training transformer-based language models on custom text data. It supports various transformer architectures (GPT-2, GPT-Neo, T5) and allows for experimentation with different hyperparameters. The model is trained using a multi-threaded approach for efficient data loading and preprocessing.

### Features

* **Data Extraction:** Supports .txt, .pdf, and .parquet files.
* **Parquet Storage:** for smaller and faster accessibility to trained data.
* **Data Cleaning:** Normalizes text, removes extraneous characters, and filters out overly short or long sentences.
* **Tokenization:** Utilizes the Hugging Face `AutoTokenizer` for subword tokenization (BPE).
* **Transformer Models:** Choose from GPT-2, GPT-Neo, or T5 architectures.
* **Hyperparameter Tuning:** Easily adjust embedding size, attention heads, layers, etc.
* **Mixed Precision:**  Leverages `torch.bfloat16` on supported GPUs for faster training.
* **Performance Evaluation:**  Tracks validation loss and perplexity.
* **Text Generation:** Generates sample text periodically during training to assess progress.
* **PyTorch Profiler:** Integrated for performance optimization.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd MyLLM
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Prepare Your Data:**
   - Place your `.txt`, `.pdf`, or `.parquet` data files in the same directory as the script or provide the full file path.

2. **Train the Model:**
   ```bash
   python train_language_model.py <data_file> --model_type <gpt2|gpt_neo|t5>
   ```
   * Replace `<data_file>` with the path to your training data.
   * Optionally specify `--model_type` to choose the desired transformer architecture (default is GPT-2).

3. **Output:**
   - The model will be trained for the specified number of iterations.
   - Validation loss and perplexity will be printed periodically.
   - Sample generated text will be shown every 100 iterations.
   - The trained model will be saved in the "saved_models" directory.
   - TensorBoard logs will be written to the "./logdir" directory, use `tensorboard --logdir=./logdir` to visualize the results in a web browser.

### Customization

* Modify the hyperparameters at the beginning of the `train_language_model.py` file to experiment with different model configurations.
* Adjust data cleaning and preprocessing steps in the `extract_text` and `clean_text` functions to suit your specific data format.
