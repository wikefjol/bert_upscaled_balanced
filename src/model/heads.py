import torch
import torch.nn as nn

class SingleClassHead(nn.Module):
    def __init__(self, in_features, hidden_layer_size, out_features, dropout_rate=0.1):
        """
        A wrapper for a predefined nn.Sequential layout:
        - Linear(input_size, hidden_size)
        - ReLU
        - Dropout
        - Linear(hidden_size, output_size)

        Args:
        - input_size (int): Size of the input features.
        - hidden_size (int): Size of the hidden layer.
        - output_size (int): Size of the output features.
        - dropout_rate (float): Dropout probability.
        """

        self.in_features = in_features
        self.hidden_layer_size = hidden_layer_size
        self.out_features = out_features
        self.dropout_rate = dropout_rate

        super(SingleClassHead, self).__init__()
        
        self.sequential = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_layer_size, self.out_features)
        )

    def forward(self, x):
        return self.sequential(x)
    
class MLMHead(nn.Module):
    def __init__(self, in_features, hidden_layer_size, out_features, dropout_rate=0.1):
        """
        A wrapper for a predefined nn.Sequential layout:
        - Linear(input_size, hidden_size)
        - ReLU
        - Dropout
        - Linear(hidden_size, output_size)

        Args:
        - input_size (int): Size of the input features.
        - hidden_size (int): Size of the hidden layer.
        - output_size (int): Size of the output features.
        - dropout_rate (float): Dropout probability.
        """

        self.in_features = in_features
        self.hidden_layer_size = hidden_layer_size
        self.out_features = out_features
        self.dropout_rate = dropout_rate

        super(MLMHead, self).__init__()
        
        self.sequential = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_layer_size, self.out_features)
        )

    def forward(self, x):
        return self.sequential(x)
    

class HierarchicalClassificationHead(nn.Module):
    def __init__(self, in_features, class_sizes, dropout_rate=0.3):
        """
        Args:
            input_dim (int): Dimensionality of the encoder's latent representation.
            class_sizes (list): Number of classes at each hierarchical level.
            dropout_rate (float): Dropout rate used in the classifier layers.
        """
        super(HierarchicalClassificationHead, self).__init__()

        self.in_features = in_features
        self.class_sizes = class_sizes
        self.num_levels = len(class_sizes)

        self.classification_heads = nn.ModuleList()

        # Create classification layers for each level
        for i, class_size in enumerate(class_sizes):
            if i == 0:
                self.classification_heads.append(
                    nn.Sequential(
                        nn.Linear(in_features, in_features // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(in_features // 2, class_size)
                    )
                )
            else:

                lvl_in_dim = in_features + class_sizes[i-1]
                lvl_intermediate_dim = lvl_in_dim //2
                lvl_out_dim = class_sizes[i]

                self.classification_heads.append(
                    nn.Sequential(
                        nn.Linear(lvl_in_dim, lvl_intermediate_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(lvl_intermediate_dim, lvl_out_dim)
                    )
                )

    def forward(self, latent_repr):
        """
        Forward pass through hierarchical classification heads.

        Args:
            latent_repr (torch.Tensor): Encoder output of shape (batch_size, input_dim).

        Returns:
            List[torch.Tensor]: A list of classification logits for each hierarchical level.
        """
        outputs = []
        current_input = latent_repr

        for head in self.classification_heads:
            logits = head(current_input)  # Get predictions for current level
            outputs.append(logits)
            current_input = torch.cat((latent_repr, logits), dim=1)  # Concatenate logits with the latent representation

        return outputs  # List of tensors with shape (batch_size, num_classes) per level