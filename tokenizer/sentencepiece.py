"""
sentencepiece.py

This script handles text tokenization using SentencePiece, a subword tokenizer designed  
for efficient and flexible tokenization of raw text data. It supports training a new  
SentencePiece model and encoding text into token IDs.

Key Features:
- **SentencePiece Model Training:** Trains a byte-pair encoding (BPE) tokenizer on the provided text dataset.  
- **Custom Special Tokens:** Includes '<user>,<bot>,<eos>' for the time being, but we can also  
  extend this set with additional tokens like '<bos>' (beginning of sentence) and '<system>'.  
- **Efficient Tokenization:** Converts text into token IDs suitable for input into a transformer model.

Optional Enhancements:
- **JSON DataLoader Support:** Currently, the script processes text from a standard file format.  
  However, we can extend it to support JSON-based conversation datasets, allowing structured  
  message storage with user/assistant roles and timestamps.

This script is used as a preprocessing step before training the TinyLLM model.
"""


import sentencepiece as spm

def train_tokenizer(texts, model_prefix='tokenizer'):
    with open('texts.txt', 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text.strip().replace('\n', ' ') + '\n')
    vocab_size = 8923
    user_defined_symbols = '<user>,<bot>,<eos>'
    spm.SentencePieceTrainer.Train(
        f'--input=texts.txt --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe --user_defined_symbols={user_defined_symbols}'
    )
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    return sp
