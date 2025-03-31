
import random
from src.utils.logging_utils import with_logging

class FrontStrategy:
   
    def __init__(self, optimal_length):
        self.optimal_length = optimal_length

    @with_logging(level=9)
    def execute(self, seq: list[list[str]]) -> list[list[str]]:
        return seq[-self.optimal_length:]
    
class EndStrategy:
   
    def __init__(self, optimal_length):
        self.optimal_length = optimal_length

    @with_logging(level=9)
    def execute(self, seq: list[list[str]]) -> list[list[str]]:
        return seq[:self.optimal_length]
    
class SlidingwindowStrategy:
   
    def __init__(self, optimal_length):
        self.optimal_length = optimal_length

    @with_logging(level=9)
    def execute(self, seq: list[list[str]]) -> list[list[str]]:
        truncated_seq = seq[:]

        # Return the whole sequence if it's already within optimal length
        if len(truncated_seq) <= self.optimal_length:
            return truncated_seq  
    
        max_start_index = len(truncated_seq) - self.optimal_length
        
        # Choose a random start index in the valid range
        start_index = random.randint(0, max_start_index)
        
        # Extract the subarray of length optimal_length
        return truncated_seq[start_index:start_index + self.optimal_length]