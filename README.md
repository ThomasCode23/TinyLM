# TinyLM

TinyLM is an experimental project aimed at creating a small, efficient language model capable of basic conversational inference. The goal is to train a coherent language model on modest hardware and with limited data, making it accessible for research, education, and resource-constrained applications. This repository contains the code and documentation for training and deploying TinyLM inference. **Bring your own dataset**.

## Overview

This project explores the practicalities of building and training a minimal Large Language Model (LLM). We focus on achieving reasonable conversational abilities with limited computational resources and datasets. This is a research-focused endeavor, and we encourage contributions and experimentation.

## Repository Structure

The project is organized as follows:

*   `tiny_llm_trainer.py`: The main training script.  This script orchestrates the data loading, tokenization, model initialization, and training process.
*   `config.py`: Contains the configuration parameters for training (e.g., data paths, hyperparameters).  Modify this file to adjust the training process.
*   `data/`: Contains training data. Place your datasets here.
*   `models/`: Contains the definitions for the TinyLM model architecture.
    *   `transformer.py`: Definition of the TransformerBlock.
    *   `positional_encoding.py`: Definition of the PositionalEncoding.
    *   `tiny_llm.py`:  Definition of the TinyLLM model.
*   `tokenizers/`: Contains code related to the SentencePiece tokenizer.
    *   `sentencepiece_tokenizer.py`:  Functions for training and using the SentencePiece tokenizer.
*   `datasets/`: Contains code for creating PyTorch datasets.
    *   `text_dataset.py`:  Definition of the `TextDataset` class.
*   `trainers/`: Contains the training loop logic.
    *   `trainer.py`:  The `train_model` function.
*   `utils/`:  Utility functions.
    *   `data_utils.py`: Contains data loading utilities (e.g., `load_conversation_data`).
*   `checkpoints/`: (Created during training) This directory will contain saved model checkpoints.
*   `requirements.txt`: Lists the Python dependencies for the project.

## Model Architecture

TinyLM utilizes a decoder-only Transformer architecture. Key components are as follows:

*   **Embedding Layer:** Converts input token IDs into dense vector representations. The embedding dimension is controlled by the `d_model` parameter (defined in `config.py`).
*   **Positional Encoding:** Adds positional information to the embeddings, allowing the model to understand the order of tokens in a sequence. Uses sinusoidal positional encodings.
*   **Transformer Blocks:** Stacked layers of self-attention and feedforward networks. Each block includes:
    *   **Multi-Head Attention:** Allows the model to attend to different parts of the input sequence simultaneously.
    *   **Layer Normalization:** Stabilizes training and improves performance.
    *   **Feedforward Network:** A two-layer feedforward network with a GELU activation function.
    *   **Dropout:** Regularization technique to prevent overfitting.
*   **Final Layer Normalization:** Applies layer normalization to the output of the final transformer block.
*   **Output Layer:** A linear layer that maps the final hidden state to the vocabulary size, producing logits for each token.
*   **Causal Masking:** During the forward pass, a causal mask is applied to the attention mechanism. This ensures that the model can only attend to tokens that come before it in the sequence.

## Training Methods and Data Considerations

TinyLM's performance is heavily influenced by the training data and methodology. We've found that a two-stage training process often yields the best results:

1.  **Pre-training on Unformatted Corpus:** Begin by training the model on a large, unformatted corpus of text (e.g., raw text from books, articles, or websites). This initial phase allows the model to learn general language patterns and build a foundational understanding of grammar and syntax. This pre-training corpus does not need to include the `<user>` and `<assistant>` special tokens.

2.  **Fine-tuning on Formatted Dataset:** After the pre-training phase, fine-tune the model on your provided dataset containing examples formatted with special tokens like `<user>` and `<assistant>`. This step teaches the model to recognize and respond appropriately within the conversational context defined by these tokens. There are example datasets included for both txt and json. Ensure you alter the data loader to accomodate any chosen tokens or formats.  The `load_conversation_data` function in `utils/data_utils.py` is responsible for loading and formatting the data.

## Vocabulary Size and Tokenization (SentencePiece)

TinyLM uses the SentencePiece tokenizer (implemented in `tokenizers/sentencepiece_tokenizer.py`). The training script automatically trains a SentencePiece model on your provided dataset using Byte Pair Encoding (BPE).

*   The vocabulary size is controlled by the `vocab_size` parameter in `config.py`.
*   Special tokens `<bos>` (beginning of sequence), `<eos>` (end-of-sequence), `<user>`, `<assistant>`, and `<system>` are added to the vocabulary.
*   It's highly recommended to keep the `vocab_size` parameter consistent between training and inference. If you retrain the tokenizer, ensure the `vocab_size` in the model configuration matches the new tokenizer's vocabulary size. The training script includes logic to handle vocabulary size changes when loading a checkpoint, but inconsistencies can still lead to unexpected behavior.

## Crafting Effective Datasets

The quality of the dataset is paramount. Here are some key considerations:

*   **Contextual Overlap:** Subtly link topics between different examples. This helps the model learn relationships and generalize better. For instance, if one example discusses "programming languages," another might touch on "software development," and a third on "artificial intelligence." This interwoven structure allows the model to connect these related concepts.
*   **Variety:** Include diverse sentence structures and conversational styles to improve the model's robustness.
*   **Realistic Conversations:** Aim to emulate realistic human-computer interactions in your training data. Use the `<user>` and `<assistant>` tokens consistently to delineate turns in the conversation.

## Model Configuration and Scalability

The model's architecture is based on a Transformer network. The core configuration parameters, now found in `config.py`, directly impact the model's performance and resource requirements. Here are the key parameters:

*   `vocab_size`: The size of the vocabulary.
*   `d_model`: Embedding dimension and attention head size.
*   `nhead`: Number of attention heads.
*   `num_layers`: Number of transformer layers.
*   `dim_feedforward`: Dimension of the feedforward network in the transformer block.
*   `dropout`: Dropout rate for regularization.

## **Getting Started**

1.  **Clone the repository:** `
2.  **Install dependencies:** `
3.  **Prepare your dataset:** Place your training data in the `data/` directory.  Adjust the `data_path` parameter in `config.py` accordingly.
4.  **Configure training parameters:** Modify the parameters in `config.py` as needed.
5.  **Run the training script:** `python tiny_llm_trainer.py`

