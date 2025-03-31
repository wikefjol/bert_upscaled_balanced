import torch
import numpy as np

from torch.utils.data import Dataset
from src.preprocessing.preprocessor import Preprocessor


class MLMDataset(Dataset):
    def __init__(self, df, preprocessor, masking_percentage):
        """
        Args:
            df (list or array-like): The dataset of sequences.
            preprocessor: An object that can process (tokenize) text -> list of token IDs.
            masking_weights (tuple or list): The probability thresholds for the types of masking.
                Example: (p_mask, p_mask + p_random, 1.0)
                i.e., if random < masking_weights[0], mask token
                      elif random < masking_weights[1], random token
                      else do nothing
            masking_percentage (float): Probability that a non-special token *attempts* to be replaced
                                        by [MASK] or random.
        """
        self.df = df
        self.preprocessor = preprocessor
        self.masking_percentage = masking_percentage
        self.ignore_index = -100

    def __len__(self):
        return len(self.df)

    def _mask(self, input_seq):
        """
        Perform in-place masking (and label creation) for MLM.
        Returns:
            masked_seq: The sequence with some tokens replaced by [MASK].
            labels: An array with original token IDs at masked positions, and -100 at unmasked positions.
        """
        seq = input_seq.copy()
        labels = seq.copy()

        # Identify maskable tokens (excluding special tokens)
        special_tokens = self.preprocessor.vocab.get_special_tokens()
        mask_candidates = np.array([1 if token not in special_tokens else 0 for token in seq])

        # Randomly generate values to decide masking
        probability_array = np.random.rand(len(seq))

        # Apply masking
        for idx in range(len(seq)):
            if mask_candidates[idx] == 0 or probability_array[idx] >= self.masking_percentage:
                labels[idx] = self.ignore_index  # Mark as non-masked
                continue

            seq[idx] = self.preprocessor.vocab.get_id("MASK")

        return seq, labels

    def _create_attention_mask(self, input_seq):
        """
        Returns an attention mask for the input sequence:
         - 1 where token != PAD
         - 0 where token == PAD
        """
        pad_id = self.preprocessor.vocab.get_id("PAD")
        attention_mask = [1 if token != pad_id else 0 for token in input_seq]
        return attention_mask

    def _add_special_tokens(self, seq):
        """
        Adds [CLS] and [SEP] tokens to the sequence.

        Args:
            seq (list): A list of token IDs representing the sequence.

        Returns:
            list: The sequence with [CLS] and [SEP] tokens added.
        """
        cls_token_id = self.preprocessor.vocab.get_id("CLS")
        sep_token_id = self.preprocessor.vocab.get_id("SEP")

        # Add [CLS] at the start and [SEP] at the end
        return [cls_token_id] + seq + [sep_token_id]

    def __getitem__(self, idx):
        """
        Returns:
        input_ids_tensor: The masked input_ids (list of token IDs as a tensor).
        mlm_labels: The MLM labels (list of token IDs or -100 for unmasked positions as a tensor).
        attention_mask_tensor: The attention mask for the sequence as a tensor.
        """
        seq = self.df.iloc[idx]
        # Convert raw item to a list of token IDs
        preprocessed_seq = self.preprocessor.process(seq)

        # Add special tokens to finalize the sequence structure
        finalized_sequence = self._add_special_tokens(preprocessed_seq)

        # Create masked input and the MLM labels
        masked_input_ids, mlm_labels = self._mask(finalized_sequence)

        # Create the attention mask
        attention_mask = self._create_attention_mask(masked_input_ids)

        # Convert to torch tensors
        input_ids_tensor = torch.tensor(masked_input_ids, dtype=torch.long)
        mlm_labels_tensor = torch.tensor(mlm_labels, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        return {
            "original_seq": seq,
            "input_ids": input_ids_tensor,
            "labels": mlm_labels_tensor,
            "attention_mask": attention_mask_tensor
        }
    
class ClassificationDataset(Dataset):
    def __init__(self,
            df,
            preprocessor,
            label_encoder,
            target_column = "species"):

        self.df = df
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.target_column = target_column

    def __len__(self):
        return len(self.df)

    def _create_attention_mask(self, input_seq):
        """
        Returns an attention mask for the input sequence:
         - 1 where token != PAD
         - 0 where token == PAD
        """
        pad_id = self.preprocessor.vocab.get_id("PAD")
        attention_mask = [1 if token != pad_id else 0 for token in input_seq]
        return attention_mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sequence = row["sequence"]
        preprocessed_sequence = self.preprocessor.process(sequence)
        attention_mask = self._create_attention_mask(preprocessed_sequence)

        label = row[self.target_column]

        return {
            "original_seq": sequence,
            "input_ids": torch.tensor(preprocessed_sequence, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": self.label_encoder.encode(label),
            "encoded_label": self.label_encoder.encode(label)
        }
    
class HieraricalClassificationDataset(Dataset):
  def __init__(self,
               df, 
               preprocessor,
               label_encoders):
  
    self.df = df
    self.preprocessor = preprocessor
    self.label_encoders = label_encoders
  
  def __len__(self):
    return len(self.df)

  def _create_attention_mask(self, input_seq):
        """
        Returns an attention mask for the input sequence:
         - 1 where token != PAD
         - 0 where token == PAD
        """
        pad_id = self.preprocessor.vocab.get_id("PAD")
        attention_mask = [1 if token != pad_id else 0 for token in input_seq]
        return attention_mask

  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    sequence = row["sequence"]
    preprocessed_sequence = self.preprocessor.process(sequence)
    attention_mask = self._create_attention_mask(preprocessed_sequence)

    lvl_output =  {}
    for taxonomic_lvl, label_encoder in self.label_encoders.items(): # One encoder per level now. 
        label = row[taxonomic_lvl]
        lvl_output[taxonomic_lvl] = {"label": label, "encoded_label": label_encoder.encode(label)}


    return {
        "original_seq": sequence,
        "input_ids": torch.tensor(preprocessed_sequence, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "output": lvl_output,
    }