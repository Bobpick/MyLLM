# GPT Language Model for Text & PDF Data

This project provides a PyTorch implementation of a GPT (Generative Pre-trained Transformer) language model, designed to be trained on text and PDF documents. It incorporates modern best practices for natural language processing (NLP) and deep learning.

## Key Features & Benefits

* **Versatile Data Processing:** Handles both text (.txt) and PDF (.pdf) files, as well as Parquet (.parquet) files, allowing you to train on a variety of sources.
* **Customizable Tokenization:** Supports both character-level and subword (BPE) tokenization for optimal model performance depending on your specific data.
* **Optimized Training:** Utilizes multithreading for efficient data loading and leverages the `ReduceLROnPlateau` scheduler for adaptive learning rate adjustments.
* **State-of-the-Art Architecture:** Implements the GPT architecture with multi-head attention, layer normalization, and feed-forward layers for powerful text generation capabilities.
* **Text Generation:** The trained model can be used to generate new text given a starting prompt.
* **Easy to Extend:** The modular structure of the code makes it straightforward to add new features or modify existing components.

## How to Use

### 1. Installation

Ensure you have the following prerequisites installed:

* Python (3.6+)
* PyTorch (1.4+)
* PyPDF2
* tokenizers
* pandas

You can install the required packages using pip:

```bash
pip install torch PyPDF2 tokenizers pandas
```
### 2. Preparing Your Data

Create a `.txt`, `.pdf` (or `.parquet`) file containing the text you want your model to learn from. 

### 3. Training the Model

Run the script from your terminal, providing the path to your data file:

```bash
python your_script_name.py your_data_file.txt
```
(Replace `your_script_name.py` and `your_data_file.txt` with the actual names.)

The model will be trained for the specified number of iterations, saving checkpoints at regular intervals.

### 4. Generating Text

Once trained, you can use the `prompt_model` function to generate text given a context:

```python
prompt_model(model, encode, decode)
```
(This will generate text based on the prompt "The quick brown fox".)


## Customization

You can adjust various hyperparameters in the code to fine-tune the model's behavior:

* `batch_size`
* `block_size`
* `max_iters`
* `learning_rate`
* `eval_iters`
* ...and more

Feel free to experiment to find the optimal settings for your dataset and use case.

## Contributing

Contributions are welcome! If you have any ideas, bug fixes, or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
