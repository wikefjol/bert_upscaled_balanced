from typing import Protocol, List, Any
from src.utils.logging_utils import with_logging
from src.utils.vocab import Vocabulary
class Strategy(Protocol):
    '''Augments sequence by imitating sequencing errors'''
    def execute(self, sequence: list[str]) -> list[str]:
        """
        Parameters
        ----------
        sequence : str
            DNA sequence

        Returns
        ----------
        sequence : str
            Augmented DNA sequence
        """

class Preprocessor:
    def __init__(
        self,
        augmentation_strategy: Strategy,
        tokenization_strategy: Strategy,
        padding_strategy: Strategy,
        truncation_strategy: Strategy,
        vocab: Vocabulary = None
    ):
        self.augmentation_strategy = augmentation_strategy
        self.tokenization_strategy = tokenization_strategy
        self.padding_strategy = padding_strategy
        self.truncation_strategy = truncation_strategy
        self.vocab = vocab
        print("DEBUG: Using ordinary preprocessor")

    def process(self, sequence: str) -> List[List[str]]:
        sequence = list(sequence)  # Convert string to list of characters
        augmented_sequence: List[str] = self.augmentation_strategy.execute(sequence)
        tokenized_sentence: List[List[str]] = self.tokenization_strategy.execute(augmented_sequence)
        padded_sentence: List[List[str]] = self.padding_strategy.execute(tokenized_sentence)
        processed_sentence: List[List[str]] = self.truncation_strategy.execute(padded_sentence)
        mapped_sentence: List[List[int]] = self.vocab.map_sentence(processed_sentence)

        
        return mapped_sentence

class OverlappingPreprocessor:
    def __init__(
        self,
        k: int,
        augmentation_strategy,
        tokenization_strategy,
        padding_strategy,
        truncation_strategy,
        vocab: Vocabulary = None,
    ):
        self.k = k
        self.augmentation_strategy = augmentation_strategy
        self.tokenization_strategy = tokenization_strategy
        self.padding_strategy = padding_strategy  # Must have optimal_length attribute.
        self.truncation_strategy = truncation_strategy
        self.vocab = vocab
        print("DEBUG: Using overlapping preprocessor")

    def process(self, sequence: str) -> List[List[int]]:
        seq_list = list(sequence)

        # 1. Augmentation
        aug_seq = self.augmentation_strategy.execute(seq_list)

        # 2. Create overlapping windows
        overlaps = [aug_seq[i:] for i in range(self.k)]
        
        # 3. Tokenize each window and flatten tokens if needed.
        tokenized = [self.tokenization_strategy.execute(win) for win in overlaps]
        tokenized = [[token if isinstance(token, str) else ''.join(token)
                      for token in window] for window in tokenized]

        # 4. Even truncation across shifts (without reserving SEP tokens)
        optimal_length = self.padding_strategy.optimal_length
        num_shifts = len(tokenized)
        tokens_avail = optimal_length  # No SEP tokens now
        base_count = tokens_avail // num_shifts
        remainder = tokens_avail % num_shifts

        truncated = []
        for i, window in enumerate(tokenized):
            allowed = base_count + (1 if i < remainder else 0)
            truncated.append(window[:allowed])
    
        # 5. Concatenate shifts without SEP tokens
        combined = []
        for window in truncated:
            combined.extend(window)

        # 6. Pad at the end if needed
        if len(combined) < optimal_length:
            pad_token = self.vocab.get_token(self.vocab.special_tokens['PAD']) if self.vocab else 'PAD'
            combined.extend([pad_token] * (optimal_length - len(combined)))
        processed_sentence = [[token] for token in combined]
        # 7. Map tokens to indices
        mapped = self.vocab.map_sentence([combined])
        return mapped