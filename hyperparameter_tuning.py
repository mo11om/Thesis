"""
Hyperparameter Tuning Script

Reads tuning specification from tuning_spec.json, performs grid search,
and outputs results to log/tuning_results.csv.

Usage:
    python hyperparameter_tuning.py                  # Run full tuning
    python hyperparameter_tuning.py --dry-run        # Preview configs without training
    python hyperparameter_tuning.py --spec custom.json  # Use custom spec file
"""

import os
import sys
import json
import copy
import itertools
import argparse
import subprocess
import csv
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np


def load_tuning_spec(spec_path: str = "tuning_spec.json") -> Dict:
    """Load tuning specification from JSON file."""
    with open(spec_path, 'r') as f:
        return json.load(f)


def load_config(config_path: str = "config.json") -> Dict:
    """Load base config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_nested_value(d: Dict, key_path: str) -> Any:
    """Get value from nested dict using dot notation (e.g., 'training.learning_rates.bert')."""
    keys = key_path.split('.')
    for key in keys:
        d = d[key]
    return d


def set_nested_value(d: Dict, key_path: str, value: Any) -> None:
    """Set value in nested dict using dot notation."""
    keys = key_path.split('.')
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def generate_param_values(param_spec: Dict) -> List[Any]:
    """Generate list of values for a parameter based on its specification."""
    if param_spec["type"] == "discrete":
        return param_spec["values"]
    
    elif param_spec["type"] == "range":
        min_val = param_spec["min"]
        max_val = param_spec["max"]
        steps = param_spec["steps"]
        scale = param_spec.get("scale", "linear")
        
        if scale == "log":
            # Logarithmic scale (useful for learning rates)
            values = np.logspace(np.log10(min_val), np.log10(max_val), steps)
        else:
            # Linear scale
            values = np.linspace(min_val, max_val, steps)
        
        return values.tolist()
    
    else:
        raise ValueError(f"Unknown parameter type: {param_spec['type']}")


def generate_grid_combinations(tuning_spec: Dict) -> List[Dict[str, Any]]:
    """Generate all combinations of hyperparameters for grid search."""
    params = tuning_spec["parameters"]
    
    # Generate all values for each parameter
    param_names = list(params.keys())
    param_values = [generate_param_values(params[name]) for name in param_names]
    
    # Create all combinations
    combinations = []
    for values in itertools.product(*param_values):
        combo = {name: val for name, val in zip(param_names, values)}
        combinations.append(combo)
    
    return combinations


def create_trial_config(base_config: Dict, param_combo: Dict[str, Any]) -> Dict:
    """Create a new config dict with the specified parameter values."""
    config = copy.deepcopy(base_config)
    
    for param_path, value in param_combo.items():
        set_nested_value(config, param_path, value)
    
    return config


def ensure_log_dir(log_dir: str) -> None:
    """Check if log directory exists, create if not."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}")
    else:
        print(f"Log directory exists: {log_dir}")


def run_training_trial(config: Dict, trial_id: int, log_dir: str) -> Tuple[float, Dict]:
    """
    Run a single training trial with the given config.
    
    Returns:
        Tuple of (validation_loss, metrics_dict)
    """
    # Save temporary config for this trial
    trial_config_path = os.path.join(log_dir, f"config_trial_{trial_id}.json")
    with open(trial_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Running Trial {trial_id}")
    print(f"Config: {trial_config_path}")
    print(f"{'='*60}")
    
    # Run gate_train.py with this config
    # Option 1: Subprocess (recommended for isolation)
    try:
        # We assume gate_train.py can accept a --config argument or reads from config.json
        # For simplicity, we'll temporarily replace config.json
        
        # Backup original config
        original_config_path = "config.json"
        backup_config_path = os.path.join(log_dir, "config_backup.json")
        
        if os.path.exists(original_config_path):
            with open(original_config_path, 'r') as f:
                original_config = json.load(f)
            with open(backup_config_path, 'w') as f:
                json.dump(original_config, f, indent=4)
        
        # Write trial config as main config
        with open(original_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Run training script
        result = subprocess.run(
            [sys.executable, "gate_train.py"],
            capture_output=True,
            text=True,
            timeout=3600 * 24  # 24 hour timeout per trial
        )
        
        # Restore original config
        if os.path.exists(backup_config_path):
            with open(backup_config_path, 'r') as f:
                original_config = json.load(f)
            with open(original_config_path, 'w') as f:
                json.dump(original_config, f, indent=4)
        
        # Parse output for validation loss (you may need to adjust based on actual output)
        # This is a placeholder - you'll need to adapt based on how gate_train.py reports metrics
        val_loss = parse_validation_loss(result.stdout, result.stderr)
        
        return val_loss, {"stdout": result.stdout[-500:], "returncode": result.returncode}
        
    except subprocess.TimeoutExpired:
        print(f"Trial {trial_id} timed out!")
        return float('inf'), {"error": "timeout"}
    except Exception as e:
        print(f"Trial {trial_id} failed: {e}")
        return float('inf'), {"error": str(e)}


def parse_validation_loss(stdout: str, stderr: str) -> float:
    """
    Parse validation loss from training output.
    Adjust this function based on your actual training output format.
    """
    # Placeholder: Look for lines like "Validation Loss: 0.123"
    import re
    
    # Try to find validation loss in output
    patterns = [
        r"val_loss[:\s]+([0-9.]+)",
        r"validation loss[:\s]+([0-9.]+)",
        r"best_val_loss[:\s]+([0-9.]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, stdout + stderr, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    # If not found, return infinity
    print("Warning: Could not parse validation loss from output")
    return float('inf')


def save_results(results: List[Dict], log_dir: str, output_file: str) -> None:
    """Save tuning results to CSV file."""
    output_path = os.path.join(log_dir, output_file)
    
    if not results:
        print("No results to save.")
        return
    
    # Get all unique keys from results
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    
    fieldnames = sorted(all_keys)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Script")
    parser.add_argument("--spec", type=str, default="tuning_spec.json",
                        help="Path to tuning specification JSON file")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to base config JSON file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview configurations without running training")
    args = parser.parse_args()
    
    # Load specifications
    print("Loading tuning specification...")
    tuning_spec = load_tuning_spec(args.spec)
    base_config = load_config(args.config)
    
    # Settings
    settings = tuning_spec.get("settings", {})
    log_dir = settings.get("log_dir", "log")
    output_file = settings.get("output_file", "tuning_results.csv")
    max_trials = settings.get("max_trials", None)
    
    # Ensure log directory exists
    ensure_log_dir(log_dir)
    
    # Generate hyperparameter combinations
    print("\nGenerating hyperparameter combinations...")
    combinations = generate_grid_combinations(tuning_spec)
    
    if max_trials and len(combinations) > max_trials:
        print(f"Limiting to {max_trials} trials (from {len(combinations)} total)")
        import random
        random.shuffle(combinations)
        combinations = combinations[:max_trials]
    
    print(f"Total trials: {len(combinations)}")
    
    # Dry run mode
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN - Preview of configurations")
        print("="*60)
        for i, combo in enumerate(combinations):
            print(f"\nTrial {i+1}:")
            for param, value in combo.items():
                print(f"  {param}: {value}")
        print("\nNo training will be executed in dry-run mode.")
        return
    
    # Run trials
    results = []
    start_time = datetime.now()
    
    for trial_id, param_combo in enumerate(combinations, 1):
        trial_config = create_trial_config(base_config, param_combo)
        
        val_loss, metrics = run_training_trial(trial_config, trial_id, log_dir)
        
        # Record result
        result = {
            "trial_id": trial_id,
            "timestamp": datetime.now().isoformat(),
            "val_loss": val_loss,
            **param_combo,
            **{f"metric_{k}": v for k, v in metrics.items() if k not in ["stdout", "stderr"]}
        }
        results.append(result)
        
        # Save intermediate results
        save_results(results, log_dir, output_file)
        
        print(f"Trial {trial_id}/{len(combinations)} - Val Loss: {val_loss}")
    
    # Final summary
    end_time = datetime.now()
    print("\n" + "="*60)
    print("TUNING COMPLETE")
    print("="*60)
    print(f"Total trials: {len(results)}")
    print(f"Duration: {end_time - start_time}")
    
    if results:
        best_result = min(results, key=lambda x: x.get("val_loss", float('inf')))
        print(f"\nBest trial: {best_result['trial_id']}")
        print(f"Best val_loss: {best_result['val_loss']}")
        print("Best parameters:")
        for param in tuning_spec["parameters"].keys():
            print(f"  {param}: {best_result.get(param, 'N/A')}")
    
    print(f"\nResults saved to: {os.path.join(log_dir, output_file)}")


if __name__ == "__main__":
    main()
