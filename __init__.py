"""
SecBERT Multi-Task Learning Framework
Information Extraction from XBRL Financial Documents

Modules:
- model: Neural network architecture (BERT + task heads + gates)
- preprocess: Data loading and tokenization
- train: Training loop and loss functions
- evaluate: Evaluation and inference utilities
"""

from .model import MultiTaskModel, GateHead
from .preprocess import (
    convert_span_to_number,
    process_batch,
    process_data,
    MultiTaskIterableDataset,
    BufferedShuffleDataset,
    ProcessedIterableDataset
)
from .train import (
    set_seed,
    CB_CE_Loss,
    FocalLoss,
    hits_at_k,
    compute_loss,
    save_checkpoint,
    load_checkpoint,
    save_model_config,
    validate_model
)
from .evaluate import (
    evaluate_classification,
    evaluate_regression,
    evaluate_model,
    load_model_config,
    get_device
)

__version__ = "1.0.0"
__all__ = [
    # Model
    "MultiTaskModel",
    "GateHead",
    
    # Preprocessing
    "convert_span_to_number",
    "process_batch",
    "process_data",
    "MultiTaskIterableDataset",
    "BufferedShuffleDataset",
    "ProcessedIterableDataset",
    
    # Training
    "set_seed",
    "CB_CE_Loss",
    "FocalLoss",
    "hits_at_k",
    "compute_loss",
    "save_checkpoint",
    "load_checkpoint",
    "save_model_config",
    "validate_model",
    
    # Evaluation
    "evaluate_classification",
    "evaluate_regression",
    "evaluate_model",
    "load_model_config",
    "get_device",
]
