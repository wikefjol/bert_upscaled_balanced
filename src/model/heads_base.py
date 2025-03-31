# heads_base.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseClassificationHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

class SingleLevelClassificationHead(BaseClassificationHead):
    def __init__(self, in_features, hidden_size, num_classes, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class MultiLevelClassificationHead(BaseClassificationHead):
    def __init__(self, in_features, class_sizes, dropout_rate=0.3):
        """
        class_sizes: list of number of classes for each level, e.g., [7, 19] for two levels.
        """
        super().__init__()
        self.num_levels = len(class_sizes)
        self.heads = nn.ModuleList()
        for i, num_classes in enumerate(class_sizes):
            if i == 0:
                self.heads.append(
                    nn.Sequential(
                        nn.Linear(in_features, in_features // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(in_features // 2, num_classes)
                    )
                )
            else:
                # Increase input dimension by previous level's output size.
                new_in = in_features + class_sizes[i-1]
                self.heads.append(
                    nn.Sequential(
                        nn.Linear(new_in, new_in // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(new_in // 2, num_classes)
                    )
                )

    def forward(self, x):
        outputs = []
        current_input = x
        for head in self.heads:
            logits = head(current_input)
            outputs.append(logits)
            current_input = torch.cat((x, logits), dim=1)
        return outputs
