# /models/TextModels.py

from torch import nn

class LSTMModel(nn.Module):
    """
    A stacked LSTM model suitable for sequence classification and next-step prediction.
    This model is designed for the LEAF text benchmarks (Sentiment140, Shakespeare).
    
    The architecture follows the one used in the original FedAvg paper.
    """
    def __init__(self, configs):
        super(LSTMModel, self).__init__()
        self.vocab_size = configs.vocab_size
        self.embedding_dim = configs.embedding_dim
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.num_classes = configs.num_classes
        self.dropout = configs.dropout

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=self.embedding_dim
        )
        
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        
        # 1. Get embeddings
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        # 2. Pass embeddings to LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # self.hidden is a tuple (h_n, c_n)
        lstm_out, hidden = self.lstm(embedded)
        
        # For classification, we only need the output of the last time step.
        # For next-char prediction, this is also what we want.
        last_step_out = lstm_out[:, -1, :]
        # last_step_out shape: (batch_size, hidden_size)
        # 3. Pass through the final fully connected layer
        out = self.fc(last_step_out)
        # out shape: (batch_size, num_classes)
        
        return out