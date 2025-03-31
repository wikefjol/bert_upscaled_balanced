
import torch.nn as nn
from transformers import BertModel, BertConfig

class Bertax(nn.Module):
    def __init__(self,num_layers=8,
            num_attention_heads=4,
            hidden_size=512,
            intermediate_size=2048,
            vocab_size=69,
            max_position_embeddings=22,
            num_classes=10,
            dropout_rate=0.1):

        super(Bertax, self).__init__()

        # Initialized to pretrain mode
        self.mode = "pretrain"

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate
        )

        self.bert = BertModel(config)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2 ),
            nn.ReLU(),
            nn.Dropout(p = dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def preTrainMode(self):
        self.mode = "pretrain"

    def classifyMode(self):
        self.mode = "classify"

    def forward(self, input_ids, attention_mask = None):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
            )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        if self.mode == "pretrain":
            #USE MLM-head for pre-training
            return self.mlm_head(sequence_output)
        if self.mode == "classify":
                return self.classification_head(pooled_output)
        else:
             raise ValueError(f"Invalid mode: {self.mode}. Use 'pretrain' or 'classify'.")
        
class ModularBertax(nn.Module):
    def __init__(self,encoder, mlm_head, classification_head):

        super(ModularBertax, self).__init__()

        # Initialized to pretrain mode
        self.mode = "pretrain"
        
        self.bert = encoder
        self.mlm_head = mlm_head
        self.classification_head = classification_head

    def preTrainMode(self):
        self.mode = "pretrain"

    def classifyMode(self):
        self.mode = "classify"

    def forward(self, input_ids, attention_mask = None):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
            )
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        if self.mode == "pretrain":
            #USE MLM-head for pre-training
            return self.mlm_head(sequence_output)
        if self.mode == "classify":
                return self.classification_head(pooled_output)
        else:
             raise ValueError(f"Invalid mode: {self.mode}. Use 'pretrain' or 'classify'.")