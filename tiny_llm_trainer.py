import torch
import os
from models.tiny_llm import TinyLLM
from tokenizers.sentencepiece_tokenizer import train_tokenizer
from datasets.text_dataset import TextDataset
from torch.utils.data import DataLoader
from trainers.trainer import train_model
from utils.data_utils import load_conversation_data
import config

def main():
    args = config.Args()

    # Load and prepare data
    print("Loading training data...")
    texts = load_conversation_data(args.data_path)
    print(f"Loaded {len(texts)} conversations from training data")

    # Initialize tokenizer
    print("Training tokenizer...")
    tokenizer = train_tokenizer(texts, vocab_size=args.vocab_size)
    vocab_size = tokenizer.get_piece_size()
    print(f"Tokenizer vocabulary size: {vocab_size}")

    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = TextDataset(texts, tokenizer, seq_length=args.seq_length)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model configuration
    model_config = {
        'vocab_size': vocab_size,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout
    }

    # Initialize model
    print("Initializing model...")
    model = TinyLLM(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        dim_feedforward=model_config['dim_feedforward'],
        dropout=model_config['dropout']
    )

    # Check for existing checkpoint
    checkpoint_path = os.path.join(args.save_dir, 'tiny_llm_best.pt')
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}. Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)

        checkpoint_model_config = checkpoint['model_config']
        checkpoint_state_dict = checkpoint['model_state_dict']

        if checkpoint_model_config['vocab_size'] != model_config['vocab_size']:
            print("Vocab size has changed. Reinitializing embedding and output layers.")
            # Remove embedding and output layer weights from state_dict
            layers_to_reinit = ['embedding.weight', 'output.weight', 'output.bias']
            for layer_name in layers_to_reinit:
                if layer_name in checkpoint_state_dict:
                    del checkpoint_state_dict[layer_name]
            # Load the rest of the state_dict
            model.load_state_dict(checkpoint_state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint_state_dict)
            print("Checkpoint loaded successfully.")
    else:
        print("No existing checkpoint found. Starting training from scratch.")

    # Train the model
    print("Starting training...")
    train_model(model, model_config, train_loader, epochs=args.epochs, lr=args.learning_rate, save_dir=args.save_dir)

    # Clean up temporary files
    if os.path.exists('texts.txt'):
        os.remove('texts.txt')

if __name__ == "__main__":
    main()
