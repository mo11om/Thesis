"""
Data Preprocessing and Dataset Classes
Handles tokenization, span alignment, and dataset loading

This module provides:
- Helper functions for number conversion and span processing
- MultiTaskIterableDataset: Reads and processes JSONL files on-the-fly
- BufferedShuffleDataset: In-memory shuffling for IterableDatasets
- ProcessedIterableDataset: Loads pre-processed tensor datasets
"""

import json
import math
import locale
from concurrent.futures import ThreadPoolExecutor
from word2number import w2n
import torch
from torch.utils.data import IterableDataset, get_worker_info


def convert_span_to_number(span):
    """
    Convert a span of text to a numeric value.
    
    Handles multiple formats:
    1. Direct numeric conversion (with locale support for thousands separators)
    2. Word-to-number conversion (e.g., "five" -> 5)
    3. Returns None if conversion fails
    
    Args:
        span (str): Text span to convert
        
    Returns:
        float or None: Numeric value if conversion successful, None otherwise
    """
    span = span.strip()
    
    # Try direct numeric conversion
    try:
        # Remove thousands separators and convert
        return locale.atof(span.replace(",", ""))
    except ValueError:
        pass  # Not a standard number, try word conversion
    
    # Try word-to-number conversion
    try:
        return w2n.word_to_num(span.lower())
    except ValueError:
        pass  # Not a recognizable word number
    
    return None  # Conversion failed


def process_batch(batch, target_attrs, tokenizer, tag2id=None, time2id=None, scale2id=None,
                  standard_rare_tags=None, classification_missing_value=-100,
                  numeric_missing_value=None):
    """
    Process a batch of raw job records into tokenized training examples.
    
    For each target within each job:
    1. Tokenize context using the provided tokenizer
    2. Find token indices corresponding to target span using offset mapping
    3. Convert target attributes to class indices
    4. Handle missing values appropriately
    
    Args:
        batch (list): List of job dictionaries with 'context', 'targets', 'golds', and 'document' info
        target_attrs (list): List of target attribute names (e.g., ["tag", "time", "scale", "negative", "fact"])
        tokenizer: HuggingFace tokenizer instance
        tag2id (dict): Mapping from tag names to class indices
        time2id (dict): Mapping from time descriptions to class indices
        scale2id (dict): Mapping from scale values to class indices
        standard_rare_tags (set): Tags considered rare/noisy (default mapped to 'standard_rare')
        classification_missing_value (int): Sentinel value for missing classification targets
        numeric_missing_value (float): Sentinel value for missing numeric targets
        
    Returns:
        list: Processed training examples with keys:
            - job_id, seq_id: Identifiers
            - context: Full tokenized context
            - input_ids, attention_mask: Tokenizer outputs
            - start_token, end_token: Target span indices
            - value: Numeric value of span (via convert_span_to_number)
            - tag, time, scale, negative: Class indices (or sentinel values)
            - fact: Numeric target value (or sentinel)
            - doc_link: Document reference
    """
    if numeric_missing_value is None:
        numeric_missing_value = torch.finfo(torch.float32).max
    
    if standard_rare_tags is None:
        standard_rare_tags = set()
    
    batch_results = []
    
    for job in batch:
        # ============ Prepare Context ============
        context_p = job['context'].get("context_p", "")
        context_t = job['context'].get("context_t", "")
        context_n = job['context'].get("context_n", "")
        
        # Construct document info
        document_info = (
            f"{job['document']['document_type']};"
            f"{job['document']['period_end_date']};"
            f"{job['document']['fiscal_year']};"
            f"{job['document']['period_focus']}"
        )
        
        # Combine contexts: table + doc_info + paragraph + note
        full_context = f"{context_t} [SEP] {document_info} [SEP] {context_p} [SEP] {context_n}"

        # ============ Tokenization ============
        tokenized = tokenizer(
            full_context,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        token_ids = tokenized["input_ids"].squeeze(0)
        offset_mapping = tokenized["offset_mapping"].squeeze(0)

        # ============ Process Targets ============
        for i, target in enumerate(job['targets']):
            start_char, end_char = target['start_pos'], target['end_pos']
            start_token, end_token = -1, -1

            # Find token indices corresponding to character span
            for idx, (start, end) in enumerate(offset_mapping):
                if start <= start_char < end:
                    start_token = idx
                if start < end_char <= end:
                    end_token = idx
                    break
            
            # Skip if span alignment failed
            if start_token == -1 or end_token == -1:
                continue
            
            # ============ Initialize Target Data ============
            target_data = {
                "job_id": job["job_id"],
                "seq_id": target["seq_id"],
                "context": full_context,
                "input_ids": token_ids,
                "attention_mask": tokenized['attention_mask'].squeeze(0),
                "start_token": start_token,
                "end_token": end_token,
                "value": convert_span_to_number(target['text']),
                "doc_link": job['document']['document_link'],
            }
            
            # Initialize all attributes with missing value sentinels
            for attr in target_attrs:
                if attr in ["tag", "time", "scale", "negative"]:
                    target_data[attr] = classification_missing_value
                elif attr == "fact":
                    target_data[attr] = numeric_missing_value

            # ============ Map Gold Labels to Indices ============
            gold_values = job['golds'][i]['value']
            for attr_idx, attr in enumerate(target['attribute']):
                value = gold_values[attr_idx]
                
                if attr == 'tag':
                    # Map rare tags to 'standard_rare'
                    if value in standard_rare_tags:
                        value = 'standard_rare'
                    target_data['tag'] = tag2id.get(value, classification_missing_value) if tag2id else classification_missing_value
                    
                elif attr == 'time':
                    target_data['time'] = time2id.get(value, classification_missing_value) if time2id else classification_missing_value
                    
                elif attr == 'fact':
                    if value:
                        target_data['fact'] = float(value)
                        target_data['negative'] = 1 if float(value) < 0 else 0
                    else:
                        target_data['fact'] = numeric_missing_value
                        target_data['negative'] = classification_missing_value
                    
                elif attr == 'scale':
                    target_data['scale'] = scale2id.get(value, classification_missing_value) if scale2id else classification_missing_value

            batch_results.append(target_data)
    
    return batch_results


def process_batch_wrapper(args):
    """
    Wrapper for process_batch to support parallel processing.
    
    Args:
        args (tuple): (batch, target_attrs, tokenizer, tag2id, time2id, scale2id, 
                       standard_rare_tags, classification_missing_value, numeric_missing_value)
    
    Returns:
        list: Processed batch results
    """
    batch, target_attrs, tokenizer, tag2id, time2id, scale2id, standard_rare_tags, \
        classification_missing_value, numeric_missing_value = args
    
    return process_batch(
        batch, target_attrs, tokenizer,
        tag2id, time2id, scale2id,
        standard_rare_tags,
        classification_missing_value,
        numeric_missing_value
    )


def process_data(data, target_attrs, tokenizer, tag2id, time2id, scale2id,
                 standard_rare_tags=None, classification_missing_value=-100,
                 numeric_missing_value=None, batch_size=32, num_workers=8):
    """
    Process a list of job records using parallel batching.
    
    Args:
        data (list): List of job dictionaries
        target_attrs (list): Target attribute names
        tokenizer: Tokenizer instance
        tag2id, time2id, scale2id (dict): Label-to-index mappings
        standard_rare_tags (set): Rare tags to consolidate
        classification_missing_value (int): Missing value sentinel
        numeric_missing_value (float): Missing numeric value sentinel
        batch_size (int): Batch size for processing
        num_workers (int): Number of worker threads
        
    Returns:
        list: All processed training examples
    """
    if numeric_missing_value is None:
        numeric_missing_value = torch.finfo(torch.float32).max
    
    if standard_rare_tags is None:
        standard_rare_tags = set()
    
    # Calculate number of batches
    num_batches = math.ceil(len(data) / batch_size)
    
    # Split data into batches
    batches = [
        data[i * batch_size: (i + 1) * batch_size] 
        for i in range(num_batches)
    ]

    # Prepare task arguments
    task_args = [
        (batch, target_attrs, tokenizer, tag2id, time2id, scale2id,
         standard_rare_tags, classification_missing_value, numeric_missing_value)
        for batch in batches
    ]

    # Process batches in parallel
    inputs = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(process_batch_wrapper, task_args):
            inputs.extend(result)
    
    return inputs


class MultiTaskIterableDataset(IterableDataset):
    """
    IterableDataset that reads JSONL files and processes them on-the-fly.
    
    Supports distributed data loading with automatic sharding across worker processes.
    Each worker reads a disjoint subset of lines from each file.
    
    Usage:
        dataset = MultiTaskIterableDataset(
            files=['train.jsonl', 'valid.jsonl'],
            target_attrs=['tag', 'time', 'scale', 'negative'],
            tokenizer=tokenizer,
            tag2id=tag2id,
            ...
        )
    """
    
    def __init__(self, files, target_attrs, tokenizer, tag2id=None, time2id=None,
                 scale2id=None, standard_rare_tags=None, 
                 classification_missing_value=-100, numeric_missing_value=None,
                 batch_size=32, num_workers=8):
        """
        Args:
            files (list): List of JSONL file paths
            target_attrs (list): Target attributes to extract
            tokenizer: HuggingFace tokenizer
            tag2id, time2id, scale2id (dict): Label mappings
            standard_rare_tags (set): Tags to consolidate
            classification_missing_value (int): Missing value sentinel
            numeric_missing_value (float): Missing numeric value sentinel
            batch_size (int): Processing batch size
            num_workers (int): Number of worker threads
        """
        self.files = files
        self.target_attrs = target_attrs
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.time2id = time2id
        self.scale2id = scale2id
        self.standard_rare_tags = standard_rare_tags or set()
        self.classification_missing_value = classification_missing_value
        self.numeric_missing_value = numeric_missing_value or torch.finfo(torch.float32).max
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def _get_sharded_lines(self, file_path):
        """
        Read lines from file with automatic sharding across worker processes.
        
        Each worker reads every nth line (where n = number of workers).
        This ensures no data duplication and complete coverage.
        
        Args:
            file_path (str): Path to JSONL file
            
        Yields:
            str: JSON line strings
        """
        worker_info = get_worker_info()
        if worker_info is None:
            # Single process mode: read all lines
            start, step = 0, 1
        else:
            # Multi-process mode: each worker reads subset
            start, step = worker_info.id, worker_info.num_workers

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % step == start:  # Each worker reads different lines
                    yield line
    
    def __iter__(self):
        """
        Iterate over JSONL files, reading and processing lines on-the-fly.
        
        Yields:
            dict: Processed training examples
        """
        for file_path in self.files:
            for line in self._get_sharded_lines(file_path):
                raw_data = json.loads(line)
                processed_data = process_batch(
                    [raw_data], 
                    self.target_attrs, 
                    self.tokenizer,
                    self.tag2id,
                    self.time2id,
                    self.scale2id,
                    self.standard_rare_tags,
                    self.classification_missing_value,
                    self.numeric_missing_value
                )
                for item in processed_data:
                    yield item


class BufferedShuffleDataset(IterableDataset):
    """
    Adds in-memory shuffling to an IterableDataset using a buffer.
    
    Maintains a buffer of samples and randomly yields from it,
    refilling from the source dataset as the buffer depletes.
    
    Trade-off: More memory usage, better shuffling quality.
    
    Usage:
        shuffled_dataset = BufferedShuffleDataset(raw_dataset, buffer_size=8000)
    """

    def __init__(self, dataset, buffer_size=8000):
        """
        Args:
            dataset (IterableDataset): Source dataset to shuffle
            buffer_size (int): Number of samples to hold in memory buffer
        """
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        """
        Yield samples with shuffling via buffer.
        
        Yields:
            dict: Samples from the buffer
        """
        import random
        
        buffer = []
        for sample in self.dataset:
            buffer.append(sample)
            if len(buffer) >= self.buffer_size:
                # Shuffle and yield half the buffer
                random.shuffle(buffer)
                while len(buffer) > self.buffer_size // 2:
                    yield buffer.pop()
        
        # Shuffle and yield remaining samples
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()


class ProcessedIterableDataset(IterableDataset):
    """
    Loads pre-processed tensor datasets (e.g., saved as JSONL with tensor data).
    
    Use this when you've already pre-processed data and saved it in a format
    that includes tokenized inputs and target tensors.
    
    Usage:
        dataset = ProcessedIterableDataset(
            files=['processed_train.jsonl'],
            num_workers=4
        )
    """

    def __init__(self, files, num_workers=4):
        """
        Args:
            files (list): List of pre-processed JSONL file paths
            num_workers (int): Number of worker threads for parallel loading
        """
        self.files = files
        self.num_workers = num_workers
    
    def _get_sharded_lines(self, file_path):
        """
        Read lines with worker sharding.
        
        Args:
            file_path (str): Path to JSONL file
            
        Yields:
            str: JSON line strings
        """
        worker_info = get_worker_info()
        if worker_info is None:
            start, step = 0, 1
        else:
            start, step = worker_info.id, worker_info.num_workers

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % step == start:
                    yield line
    
    def __iter__(self):
        """
        Iterate over pre-processed JSONL files.
        
        Yields:
            dict: Pre-processed training example
        """
        for file_path in self.files:
            for line in self._get_sharded_lines(file_path):
                try:
                    data = json.loads(line)
                    # Convert lists to tensors if needed
                    if isinstance(data.get('input_ids'), list):
                        data['input_ids'] = torch.tensor(data['input_ids'])
                    if isinstance(data.get('attention_mask'), list):
                        data['attention_mask'] = torch.tensor(data['attention_mask'])
                    yield data
                except (json.JSONDecodeError, TypeError):
                    # Skip malformed lines
                    continue
