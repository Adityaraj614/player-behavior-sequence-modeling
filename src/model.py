import torch 
import torch.nn as nn

class PlayerBehaviourLSTM(nn.Module):
    def __init__(
            self,
            num_actions,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
    ):
        super().__init__()

        self.embedding =nn.Embedding(
            num_embeddings=num_actions +1,
            embedding_dim = embedding_dim,
            padding_idx=0
        )

        self.lstm =nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        
        embedded =self.embedding(x)
        outputs, (hidden, cell) =self.lstm(embedded)

        final_hidden =hidden[-1]
        return final_hidden


class PlayerBehaviorNextAction(nn.Module):
    def __init__(self, num_actions, embedding_dim=32, hidden_dim=64):
        super().__init__()

        self.backbone = PlayerBehaviourLSTM(
            num_actions=num_actions,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )

        self.classifier = nn.Linear(hidden_dim, num_actions + 1)  # +1 for padding

    def forward(self, x):
        h = self.backbone(x)          # [batch, hidden_dim]
        logits = self.classifier(h)   # [batch, num_actions+1]
        return logits
