import importlib

from inspect import signature
from typing import Any, Protocol


from src.preprocessing.augmentation import SequenceModifier
from src.preprocessing.preprocessor import Preprocessor, OverlappingPreprocessor
from src.utils.errors import ConstructionError, StrategyError
from src.utils.logging_utils import with_logging

class Modifier(Protocol):
    """Protocol defining sequence modification methods."""
    alphabet: list[str]

    def _insert(self, seq: list[str], idx: int) -> None: pass
    def _replace(self, seq: list[str], idx: int) -> None: pass
    def _delete(self, seq: list[str], idx: int) -> None: pass
    def _swap(self, seq: list[str], idx: int) -> None: pass

def load_strategy_module(strategy_type: str) -> Any:
    """Dynamically load and return the strategy module."""
    full_module_name = f"src.preprocessing.{strategy_type}"
    try:
        return importlib.import_module(full_module_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module '{full_module_name}' could not be imported.")

def prepare_strategy(module: Any, class_name: str, **kwargs) -> Any:
    """Validate and instantiate the strategy class from the module."""
    try:
        strategy_class = getattr(module, class_name)
    except AttributeError:
        available_classes = [attr for attr in dir(module) if not attr.startswith("_")]
        raise StrategyError(
            f"Class '{class_name}' not found in module '{module.__name__}'. "
            f"Available classes: {available_classes}"
        )

    init_signature = signature(strategy_class)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_signature.parameters}

    missing_args = [
        param.name for param in init_signature.parameters.values()
        if param.default == param.empty and param.name not in filtered_kwargs
    ]
    if missing_args:
        raise ValueError(f"Missing required arguments for {strategy_class.__name__}: {missing_args}")

    return strategy_class(**filtered_kwargs)

def get_strategy(strategy_type: str, **kwargs) -> Any:
    """Dynamically load and return an instance of a strategy class."""
    strategy_name = kwargs.pop("strategy", None)
    if not strategy_name:
        raise ValueError(f"Missing 'strategy' in configuration for {strategy_type}.")
    
    class_name = strategy_name.capitalize() + "Strategy"
    module = load_strategy_module(strategy_type)
    return prepare_strategy(module, class_name, **kwargs)

def create_preprocessor(config, vocab, training=True):
    # Build shared strategies:
    augmentation_config = config["augmentation"]["training"] if training else config["augmentation"]["evaluation"]
    tokenization_config = config["tokenization"]
    padding_config = config["padding"]
    truncation_config = config["truncation"]

    modifier = SequenceModifier(vocab.get_alphabet())
    augmentation_strategy = get_strategy("augmentation", modifier=modifier, **augmentation_config)
    tokenization_strategy = get_strategy("tokenization", **tokenization_config)
    padding_strategy = get_strategy("padding", **padding_config)
    truncation_strategy = get_strategy("truncation", **truncation_config)

    if config["tokenization"].get("overlapping", False):
        k = tokenization_config.get("k")
        return OverlappingPreprocessor(
            k=k,
            augmentation_strategy=augmentation_strategy,
            tokenization_strategy=tokenization_strategy,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            vocab=vocab,
        )
    else:
        print("Using ordinary preprocssor:")
        return Preprocessor(
            augmentation_strategy=augmentation_strategy,
            tokenization_strategy=tokenization_strategy,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            vocab=vocab,
        )
