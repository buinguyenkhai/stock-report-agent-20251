"""

Usage:
    python run_benchmark.py                     # Run all tasks with models from config
    python run_benchmark.py --task item_matching  # Run specific task
    python run_benchmark.py --models "gemini-2.0-flash,meta-llama/llama-3.3-70b-instruct:free"
    python run_benchmark.py --skip-validation   # Skip structured output validation
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.model_benchmark import (
    ModelBenchmark,
    BenchmarkTask,
    print_benchmark_summary
)
from logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = "benchmark_config.json"


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from JSON file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_models_from_config(config: dict) -> list:
    """Extract all models from config file."""
    models_config = config.get("models", {})
    
    # Get OpenRouter models
    models = models_config.get("openrouter", [])
    
    return models


def get_tasks_from_config(config: dict) -> list:
    """Extract enabled tasks from config file."""
    tasks_config = config.get("tasks", {})
    enabled = tasks_config.get("enabled", ["all"])
    
    task_mapping = {
        "item_matching": BenchmarkTask.ITEM_MATCHING,
        "unit_detection": BenchmarkTask.UNIT_DETECTION,
        "all": BenchmarkTask.ALL
    }
    
    return [task_mapping.get(t, BenchmarkTask.ALL) for t in enabled]


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM model benchmarks for financial report processing tasks"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to benchmark config JSON file (default: {DEFAULT_CONFIG_PATH})"
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to test (overrides config)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["item_matching", "unit_detection", "all"],
        help="Specific task to benchmark (overrides config)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip structured output validation (test all models even if they might fail)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save benchmark results (overrides config)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output, only show summary"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    settings = config.get("settings", {})
    
    # Determine models to test
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = get_models_from_config(config)
    
    if not models:
        logger.error("No models specified! Add models to config or use --models flag")
        sys.exit(1)
    
    # Determine tasks to run
    if args.task:
        task_mapping = {
            "item_matching": BenchmarkTask.ITEM_MATCHING,
            "unit_detection": BenchmarkTask.UNIT_DETECTION,
            "all": BenchmarkTask.ALL
        }
        tasks = [task_mapping[args.task]]
    else:
        tasks = get_tasks_from_config(config)
    
    # Determine output directory
    output_dir = args.output_dir or settings.get("output_dir", "benchmark_results")
    
    # Print configuration
    if not args.quiet:
        print("\n" + "=" * 60)
        print("LLM MODEL BENCHMARK")
        print("=" * 60)
        print(f"Models to test: {len(models)}")
        for m in models:
            print(f"  - {m}")
        print(f"Tasks: {[t.value for t in tasks]}")
        print(f"Output directory: {output_dir}")
        print(f"Validate structured output: {not args.skip_validation}")
        print("=" * 60 + "\n")
    
    # Create and run benchmark
    benchmark = ModelBenchmark(
        models=models,
        ground_truth_dir=settings.get("ground_truth_dir", "data/ground_truth"),
        evaluation_results_dir=settings.get("evaluation_results_dir", "evaluation_results"),
        output_dir=output_dir
    )
    
    try:
        report = benchmark.run(
            tasks=tasks,
            validate_structured_output=not args.skip_validation
        )
        
        # Print summary
        print_benchmark_summary(report)
        
        print(f"\nDetailed results saved to: {output_dir}/latest.json")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
