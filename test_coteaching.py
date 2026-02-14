"""
Smoke test for Co-teaching loss function and forget rate schedule.

Usage:
    cd /home/mo1om/code/XBRL/model_code && python test_coteaching.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Add project path
sys.path.insert(0, '.')

from secbert_utils import compute_coteaching_loss, CLASSIFICATION_MISSING_VALUE


def test_coteaching_loss_basic():
    """Test basic co-teaching loss computation."""
    print("=" * 60)
    print("Test 1: Basic co-teaching loss computation")
    print("=" * 60)
    
    batch_size = 16
    num_tags = 10
    num_times = 2
    num_scales = 25
    
    # Simulate outputs from two models
    outputs1 = {
        "tag": torch.randn(batch_size, num_tags, requires_grad=True),
        "time": torch.randn(batch_size, num_times, requires_grad=True),
        "scale": torch.randn(batch_size, num_scales, requires_grad=True),
        "negative": torch.randn(batch_size, 2, requires_grad=True),
    }
    outputs2 = {
        "tag": torch.randn(batch_size, num_tags, requires_grad=True),
        "time": torch.randn(batch_size, num_times, requires_grad=True),
        "scale": torch.randn(batch_size, num_scales, requires_grad=True),
        "negative": torch.randn(batch_size, 2, requires_grad=True),
    }
    
    # Simulate targets (all valid)
    targets = {
        "tag": torch.randint(0, num_tags, (batch_size,)),
        "time": torch.randint(0, num_times, (batch_size,)),
        "scale": torch.randint(0, num_scales, (batch_size,)),
        "negative": torch.randint(0, 2, (batch_size,)),
    }
    
    forget_rate = 0.2
    loss1, loss2, details = compute_coteaching_loss(outputs1, outputs2, targets, forget_rate)
    
    assert loss1.requires_grad, "loss1 should be differentiable"
    assert loss2.requires_grad, "loss2 should be differentiable"
    assert loss1.item() > 0, "loss1 should be positive"
    assert loss2.item() > 0, "loss2 should be positive"
    
    # Check backward works
    loss1.backward()
    loss2.backward()
    
    print(f"  loss1 = {loss1.item():.4f}")
    print(f"  loss2 = {loss2.item():.4f}")
    print(f"  Details: {details}")
    print("  ✅ PASSED: Basic loss computation and backward pass work.\n")


def test_coteaching_loss_with_missing_labels():
    """Test that ignore_index samples are properly excluded."""
    print("=" * 60)
    print("Test 2: Co-teaching loss with missing labels (ignore_index)")
    print("=" * 60)
    
    batch_size = 16
    num_tags = 10
    
    outputs1 = {
        "tag": torch.randn(batch_size, num_tags, requires_grad=True),
        "time": torch.randn(batch_size, 2, requires_grad=True),
        "scale": torch.randn(batch_size, 25, requires_grad=True),
        "negative": torch.randn(batch_size, 2, requires_grad=True),
    }
    outputs2 = {
        "tag": torch.randn(batch_size, num_tags, requires_grad=True),
        "time": torch.randn(batch_size, 2, requires_grad=True),
        "scale": torch.randn(batch_size, 25, requires_grad=True),
        "negative": torch.randn(batch_size, 2, requires_grad=True),
    }
    
    # Half the samples have missing tag labels
    tag_targets = torch.randint(0, num_tags, (batch_size,))
    tag_targets[batch_size // 2:] = CLASSIFICATION_MISSING_VALUE
    
    targets = {
        "tag": tag_targets,
        "time": torch.randint(0, 2, (batch_size,)),
        "scale": torch.randint(0, 25, (batch_size,)),
        "negative": torch.randint(0, 2, (batch_size,)),
    }
    
    forget_rate = 0.2
    loss1, loss2, details = compute_coteaching_loss(outputs1, outputs2, targets, forget_rate)
    
    # Check that num_valid for tag is batch_size // 2
    assert details["tag"]["num_valid"] == batch_size // 2, \
        f"Expected {batch_size // 2} valid tag samples, got {details['tag']['num_valid']}"
    
    # num_remember should be (1 - 0.2) * 8 = 6
    expected_remember = int((1 - forget_rate) * (batch_size // 2))
    assert details["tag"]["num_remember"] == expected_remember, \
        f"Expected {expected_remember} remembered, got {details['tag']['num_remember']}"
    
    # Backward should work
    loss1.backward()
    loss2.backward()
    
    print(f"  Tag valid: {details['tag']['num_valid']}, remembered: {details['tag']['num_remember']}")
    print(f"  loss1 = {loss1.item():.4f}")
    print(f"  loss2 = {loss2.item():.4f}")
    print("  ✅ PASSED: Missing labels correctly handled.\n")


def test_coteaching_loss_all_missing():
    """Test behavior when all labels are missing for a task."""
    print("=" * 60)
    print("Test 3: All labels missing for a task")
    print("=" * 60)
    
    batch_size = 8
    
    outputs1 = {
        "tag": torch.randn(batch_size, 10, requires_grad=True),
        "time": torch.randn(batch_size, 2, requires_grad=True),
        "scale": torch.randn(batch_size, 25, requires_grad=True),
        "negative": torch.randn(batch_size, 2, requires_grad=True),
    }
    outputs2 = {
        "tag": torch.randn(batch_size, 10, requires_grad=True),
        "time": torch.randn(batch_size, 2, requires_grad=True),
        "scale": torch.randn(batch_size, 25, requires_grad=True),
        "negative": torch.randn(batch_size, 2, requires_grad=True),
    }
    
    # All tag labels are missing
    targets = {
        "tag": torch.full((batch_size,), CLASSIFICATION_MISSING_VALUE, dtype=torch.long),
        "time": torch.randint(0, 2, (batch_size,)),
        "scale": torch.randint(0, 25, (batch_size,)),
        "negative": torch.randint(0, 2, (batch_size,)),
    }
    
    loss1, loss2, details = compute_coteaching_loss(outputs1, outputs2, targets, 0.2)
    
    assert details["tag"] == 0.0, "Tag loss details should be 0.0 when all labels missing"
    print(f"  Tag detail: {details['tag']} (correctly 0.0 for all-missing)")
    print(f"  loss1 = {loss1.item():.4f}")
    print("  ✅ PASSED: All-missing labels handled gracefully.\n")


def test_forget_rate_schedule():
    """Test the forget rate schedule."""
    print("=" * 60)
    print("Test 4: Forget rate schedule")
    print("=" * 60)
    
    from coteaching_train import get_forget_rate
    
    noise_rate = 0.2
    num_gradual = 10
    
    rates = []
    for epoch in range(20):
        rate = get_forget_rate(epoch, noise_rate, num_gradual)
        rates.append(rate)
    
    # During warmup, rates should increase linearly
    for i in range(1, num_gradual):
        assert rates[i] >= rates[i-1], f"Rates should be non-decreasing during warmup: {rates[i]} < {rates[i-1]}"
    
    # After warmup, rates should be at noise_rate
    for i in range(num_gradual, 20):
        assert abs(rates[i] - noise_rate) < 1e-6, f"Rate should be {noise_rate} after warmup, got {rates[i]}"
    
    print(f"  Rates (first 15 epochs): {[f'{r:.3f}' for r in rates[:15]]}")
    print(f"  Warmup: linearly increasing ✅")
    print(f"  Post-warmup: constant at {noise_rate} ✅")
    print("  ✅ PASSED: Forget rate schedule is correct.\n")


def test_forget_rate_zero():
    """Test with forget_rate = 0 (should use all valid samples)."""
    print("=" * 60)
    print("Test 5: Forget rate = 0 (all samples selected)")
    print("=" * 60)
    
    batch_size = 8
    num_tags = 5
    
    outputs1 = {
        "tag": torch.randn(batch_size, num_tags, requires_grad=True),
        "time": torch.randn(batch_size, 2, requires_grad=True),
        "scale": torch.randn(batch_size, 25, requires_grad=True),
        "negative": torch.randn(batch_size, 2, requires_grad=True),
    }
    outputs2 = {
        "tag": torch.randn(batch_size, num_tags, requires_grad=True),
        "time": torch.randn(batch_size, 2, requires_grad=True),
        "scale": torch.randn(batch_size, 25, requires_grad=True),
        "negative": torch.randn(batch_size, 2, requires_grad=True),
    }
    targets = {
        "tag": torch.randint(0, num_tags, (batch_size,)),
        "time": torch.randint(0, 2, (batch_size,)),
        "scale": torch.randint(0, 25, (batch_size,)),
        "negative": torch.randint(0, 2, (batch_size,)),
    }
    
    loss1, loss2, details = compute_coteaching_loss(outputs1, outputs2, targets, forget_rate=0.0)
    
    # With forget_rate=0, all valid samples should be remembered
    assert details["tag"]["num_remember"] == batch_size, \
        f"Expected all {batch_size} samples remembered, got {details['tag']['num_remember']}"
    
    print(f"  num_remember = {details['tag']['num_remember']} (equals batch_size)")
    print("  ✅ PASSED: forget_rate=0 selects all samples.\n")


if __name__ == "__main__":
    print("\n🧪 Running Co-teaching Smoke Tests\n")
    
    test_coteaching_loss_basic()
    test_coteaching_loss_with_missing_labels()
    test_coteaching_loss_all_missing()
    test_forget_rate_schedule()
    test_forget_rate_zero()
    
    print("=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)
