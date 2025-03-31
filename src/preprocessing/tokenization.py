import random
import logging
from src.utils.logging_utils import with_logging

system_logger = logging.getLogger("system_logger")
class KmerStrategy:
    def __init__(self, k: int, padding_alphabet: list[str] = ['A','C','G','T']):
        self.k = k
        self.padding_alphabet = padding_alphabet
    
    @with_logging(level=8)
    def _make_divisible_by_k(self, input_seq: list[str]) -> list[str]:
        """Pads seq if its length is not divisible by k, using random characters from the padding alphabet."""
        seq = input_seq[:]
        while len(seq) % self.k != 0:
           seq.append(random.choice(self.padding_alphabet))
        return seq
    
    @with_logging(level=9)
    def execute(self, input_seq: list[str]) -> list[list[str]]:
        """Tokenizes the seq into k-mers, adding padding if necessary."""

        seq = input_seq[:]
        if self.k == 0:
            self.k = 1
            system_logger.warning(f"[PROCESSING.TOKENIZATION] 'KmerStrategy.execute':  k cant be {0}; set to {1}")

        remainder = len(seq) % self.k
        if remainder != 0:
            seq = self._make_divisible_by_k(seq)

        # Tokenize into k-mers
        kmer_seq = [''.join(seq[i:i + self.k]) for i in range(0, len(seq), self.k)]
        tokenized_sentence = [[kmer] for kmer in kmer_seq]

        return tokenized_sentence
    
    
    
