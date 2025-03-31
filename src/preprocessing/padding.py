
import random
from src.utils.logging_utils import with_logging

class EndStrategy:
   
    def __init__(self, optimal_length):
        self.optimal_length = optimal_length

    @with_logging(level=9)
    def execute(self, seq: list[list[str]]) -> list[list[str]]:
        # Create a copy to avoid modifying the original sequence
        padded_seq = seq[:]
        
        # Add ['PAD'] sub-lists until the length matches optimal_length
        while len(padded_seq) < self.optimal_length:
            padded_seq.append(['PAD'])
        
        # If seq is already longer than optimal_length, trim it
        return padded_seq
   
class FrontStrategy:
   
    def __init__(self, optimal_length):
        self.optimal_length = optimal_length

    @with_logging(level=9)
    def execute(self, seq: list[list[str]]) -> list[list[str]]:
        # Calculate the number of padding elements needed
        padding_needed = max(0, self.optimal_length - len(seq))
        
        # Create padding and prepend to sequence
        padded_seq = [['PAD']] * padding_needed + seq
        
        # Trim if padded sequence exceeds optimal length
        return padded_seq
    
class PRandomStrategy:
    
    def __init__(self, optimal_length):
        self.optimal_length = optimal_length

    @with_logging(level=9)
    def execute(self, seq: list[list[str]]) -> list[list[str]]:
        # Calculate the total padding needed
        padding_needed = max(0, self.optimal_length - len(seq))
        
        # Randomly decide how much padding to put in front
        front_padding_count = random.randint(0, padding_needed)
        end_padding_count = padding_needed - front_padding_count
        
        # Create the padded sequence
        padded_seq = [['PAD']] * front_padding_count + seq + [['PAD']] * end_padding_count
        
        # If seq is already longer than optimal_length, trim it
        return padded_seq