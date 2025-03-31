class PreprocessingError(Exception):
    """Base class for all preprocessing-related errors."""
    pass

class AugmentationError(PreprocessingError):
    """Raised for errors in the augmentation process."""
    pass

class TokenizationError(PreprocessingError):
    """Raised for errors in the tokenization process."""
    pass

class PaddingError(PreprocessingError):
    """Raised for errors in padding strategies."""
    pass

class TruncationError(PreprocessingError):
    """Raised for errors in truncation strategies."""
    pass

class ConstructionError(Exception):
    """Exception raised for errors in model construction."""
    def __init__(self, message="An error occurred during model construction."):
        # Pass the custom message to the base Exception class
        super().__init__(message)

class StrategyError(PreprocessingError):
    """Base class for all preprocessing-related errors."""
    def __init__(self, message="An error occurred related to strategies."):
        # Pass the custom message to the base Exception class
        super().__init__(message)