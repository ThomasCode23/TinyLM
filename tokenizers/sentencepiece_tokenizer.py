import sentencepiece as spm
import os

def train_tokenizer(texts, model_prefix='tokenizer', vocab_size=None):
    # Save all texts to a file for training SentencePiece model
    with open('texts.txt', 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text.strip().replace('\n', ' ') + '\n')

    if vocab_size is None:
        # Estimate vocab_size from data
        unique_chars = set()
        for text in texts:
            unique_chars.update(set(text))
        vocab_size = len(unique_chars) + 100  # Add buffer for subwords and special tokens
        vocab_size = max(vocab_size, 1000)    # Ensure minimum vocab size
    print(f'Setting vocab_size: {vocab_size}')

    # Define special tokens
    user_defined_symbols = '<bos>,<eos>,<user>,<assistant>,<system>'
    print(f'Using special tokens: {user_defined_symbols}')

    # Train SentencePiece model
    spm.SentencePieceTrainer.Train(
        f'--input=texts.txt \
        --model_prefix={model_prefix} \
        --vocab_size={vocab_size} \
        --model_type=bpe \
        --user_defined_symbols={user_defined_symbols}'
    )

    # Load the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    return sp
