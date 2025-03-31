import random
import logging
from src.utils.logging_utils import with_logging

class SequenceModifier():
    ''' Modifies a sequence at a specific position'''
    def __init__(self, alphabet: list[str]):
        self.alphabet = alphabet
    
    @with_logging(level=8)
    def _insert(self, seq: list[str], idx: int) -> None:
        insert_idx = random.choice([idx, idx + 1])
        if insert_idx <= len(seq):
            seq.insert(insert_idx, random.choice(self.alphabet))
    
    @with_logging(level=8)
    def _replace(self, seq: list[str], idx: int) -> None:
        seq[idx] = random.choice(self.alphabet)
    
    @with_logging(level=8)
    def _delete(self, seq: list[str], idx: int) -> None:
        if len(seq) > 1:
            seq.pop(idx)
    
    @with_logging(level=8)
    def _swap(self, seq: list[str], idx: int) -> None:
        swap_pos = idx + random.choice([-1, 1])
        if 0 <= swap_pos < len(seq):
            seq[idx], seq[swap_pos] = seq[swap_pos], seq[idx]

class BaseStrategy():
    ''' Standard augmentation strategy'''
    def __init__(self, modifier: SequenceModifier, alphabet: list[str], modification_probability: float = 0.05):
        self.alphabet = alphabet
        self.modifier = modifier
        self.modification_probability = modification_probability
        self.operation_map = {
            'insert': self.modifier._insert,
            'replace': self.modifier._replace,
            'delete': self.modifier._delete,
            'swap': self.modifier._swap
        }
        self.operations = list(self.operation_map.keys())
        self.weights = [0.25, 0.25, 0.25, 0.25]
    
    @with_logging(level=9)
    def execute(self, seq: list[str]) -> list[str]:
        augmented_seq = seq[:]
        for pos, _ in enumerate(augmented_seq):
            
            should_modify = random.random() < self.modification_probability
            if should_modify:     
                operation = random.choices(self.operations, weights=self.weights, k=1)[0]
                self.operation_map[operation](augmented_seq, pos)
        return augmented_seq
    
class RandomStrategy(BaseStrategy):
    ''' Modify at every position augmentation'''
    def __init__(self, alphabet, modifier: SequenceModifier):
        super().__init__(modifier, alphabet, modification_probability=1)

class IdentityStrategy(BaseStrategy):
    ''' Do-nothing augmentation'''
    @with_logging(level=9)
    def execute(self, seq: list[str]) -> list[str]:

        return seq[:]
        
    
    