class Args:
    def __init__(self, data_path="data/dataset.txt", epochs=100, batch_size=32, seq_length=256, learning_rate=0.001, save_dir="checkpoints", vocab_size=10200, d_model=512, nhead=8, num_layers=8, dim_feedforward=2048, dropout=0.1):
        self.data_path = data_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
