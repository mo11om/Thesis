"""
SecBERT Model Architecture
Multi-Task Learning BERT Model for Information Extraction

This module defines:
- GateHead: Task-specific gating mechanism
- MultiTaskModel: Main BERT-based model with multiple heads and gates
"""

import torch
import torch.nn as nn
from transformers import BertModel


class GateHead(nn.Module):
    """
    Multi-Task Gate Head that outputs separate noise scores for each task.
    
    The gate mechanism helps the model learn which samples are noisy for each task.
    
    Input: [CLS] token representation
    Output: Dictionary of 4 gates (one per task)
            {"tag": [0-1], "time": [0-1], "scale": [0-1], "negative": [0-1]}
            
    Each gate score represents the "noise probability":
        1 = High Noise (Ignore this sample for this task)
        0 = Clean Data (Learn from this sample for this task)
    """
    
    def __init__(self, hidden_size, dropout_prob=0.1):
        """
        Args:
            hidden_size (int): BERT's hidden dimension size (typically 768)
            dropout_prob (float): Dropout probability for regularization
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Separate output heads for each task
        self.gate_tag = nn.Linear(hidden_size, 1)
        self.gate_time = nn.Linear(hidden_size, 1)
        self.gate_scale = nn.Linear(hidden_size, 1)
        self.gate_negative = nn.Linear(hidden_size, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [CLS] token representation, shape (batch_size, hidden_size)
            
        Returns:
            Dictionary with per-task gate scores, each shape (batch_size, 1)
        """
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)  # Tanh is standard for BERT pooler/heads
        x = self.dropout(x)
        
        # Compute gate scores for each task using sigmoid (0-1 range)
        gate_tag = self.sigmoid(self.gate_tag(x))        # (batch_size, 1)
        gate_time = self.sigmoid(self.gate_time(x))
        gate_scale = self.sigmoid(self.gate_scale(x))
        gate_negative = self.sigmoid(self.gate_negative(x))
        
        return {
            "tag": gate_tag,
            "time": gate_time,
            "scale": gate_scale,
            "negative": gate_negative
        }


class MultiTaskModel(nn.Module):
    """
    Multi-Task Learning Model based on BERT with separate heads for each task.
    
    Architecture:
    - BERT encoder: Processes input text
    - Task-specific heads: Tag, Time, Scale, Negative classification heads
    - GateHead: Learns task-specific sample importance/noise scores
    
    Tasks:
    1. Tag Classification: Extract XBRL tag
    2. Time Classification: Time dimension (instant/period, past/current/future)
    3. Scale Classification: Magnitude of the number (-12 to +12)
    4. Negative Detection: Binary classification of negative values
    """
    
    def __init__(self, bert_model_name, num_tags, num_times, num_scales):
        """
        Args:
            bert_model_name (str): HuggingFace BERT model identifier
                                  e.g., "nlpaueb/sec-bert-base"
            num_tags (int): Number of tag classes
            num_times (int): Number of time classes
            num_scales (int): Number of scale classes (25 for -12 to +12)
        """
        super(MultiTaskModel, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        
        # ============ Task-Specific Classification Heads ============
        
        # Tag Head: Multiple token-level attributes for XBRL classification
        self.tag_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), 
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_tags)
        )

        # Time Head: Temporal dimension classification
        self.time_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), 
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_times)
        )

        # Scale Head: Magnitude classification (powers of 10)
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), 
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, num_scales)
        )

        # Negative Head: Binary classification for negative values
        self.negative_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), 
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, 2)  # Binary: [positive, negative]
        )
        
        # ============ Multi-Task Gate Head ============
        # Outputs 4 separate gates (one per task)
        self.gate_head = GateHead(hidden_size)
        
    def forward(self, input_ids, attention_mask, start_tokens, end_tokens):
        """
        Forward pass of the multi-task model.
        
        Args:
            input_ids (torch.LongTensor): Input token IDs, shape (batch_size, seq_length)
            attention_mask (torch.LongTensor): Attention mask, shape (batch_size, seq_length)
            start_tokens (torch.LongTensor): Start indices of target spans, shape (batch_size,)
            end_tokens (torch.LongTensor): End indices of target spans, shape (batch_size,)
            
        Returns:
            dict: Dictionary containing:
                - "tag" (torch.FloatTensor): Tag logits, shape (batch_size, num_tags)
                - "time" (torch.FloatTensor): Time logits, shape (batch_size, num_times)
                - "scale" (torch.FloatTensor): Scale logits, shape (batch_size, num_scales)
                - "negative" (torch.FloatTensor): Negative logits, shape (batch_size, 2)
                - "gates" (dict): Per-task gate scores
                    - "tag": (batch_size, 1)
                    - "time": (batch_size, 1)
                    - "scale": (batch_size, 1)
                    - "negative": (batch_size, 1)
        """
        # ============ BERT Encoding ============
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
        cls_token = sequence_output[:, 0, :]               # (batch_size, hidden_size)
        
        # ============ Target Span Aggregation ============
        # Extract and average embeddings within target span
        target_embeddings = [
            sequence_output[i, start_tokens[i]:end_tokens[i] + 1].mean(dim=0)
            for i in range(input_ids.size(0))
        ]  # List of (hidden_size,) tensors
        
        target_embeddings = torch.stack(target_embeddings)  # (batch_size, hidden_size)
        
        # ============ Task-Specific Predictions ============
        tag_logits = self.tag_head(target_embeddings)        # (batch_size, num_tags)
        time_logits = self.time_head(target_embeddings)      # (batch_size, num_times)
        scale_logits = self.scale_head(target_embeddings)    # (batch_size, num_scales)
        negative_logits = self.negative_head(target_embeddings)  # (batch_size, 2)
        
        # ============ Task-Specific Gates ============
        gates = self.gate_head(cls_token)  # Dict of 4 gates, each (batch_size, 1)
        
        return {
            "tag": tag_logits,
            "time": time_logits,
            "scale": scale_logits,
            "negative": negative_logits,
            "gates": gates  # Dict: {"tag": gate, "time": gate, "scale": gate, "negative": gate}
        }
