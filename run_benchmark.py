"""
Tests different LLM APIs on real pipeline tasks:
- Extraction: Finding and extracting financial tables from OCR markdown
- Parsing: Converting extracted content to structured JSON with correct values
- Full Pipeline: End-to-end extraction + parsing

Usage:
    python run_benchmark.py                          # Run all tasks
    python run_benchmark.py --task extraction        # Extraction only
    python run_benchmark.py --task parsing           # Parsing only
    python run_benchmark.py --task full_pipeline     # Full pipeline
    python run_benchmark.py --models "model1,model2" # Override models
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.pipeline_benchmark import (
    PipelineBenchmark,
    BenchmarkTask,
    print_benchmark_summary,
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
        "extraction": BenchmarkTask.EXTRACTION,
        "parsing": BenchmarkTask.PARSING,
        "full_pipeline": BenchmarkTask.FULL_PIPELINE,
        "all": BenchmarkTask.ALL,
    }
    
    return [task_mapping.get(t, BenchmarkTask.ALL) for t in enabled]


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM benchmarks on pipeline extraction and parsing tasks"
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
        choices=["extraction", "parsing", "full_pipeline", "all"],
        help="Specific task to benchmark (overrides config)"
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
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare OCR test data from PDFs, don't run benchmark"
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip automatic test data preparation"
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
            "extraction": BenchmarkTask.EXTRACTION,
            "parsing": BenchmarkTask.PARSING,
            "full_pipeline": BenchmarkTask.FULL_PIPELINE,
            "all": BenchmarkTask.ALL,
        }
        tasks = [task_mapping[args.task]]
    else:
        tasks = get_tasks_from_config(config)
    
    # Determine output directory
    output_dir = args.output_dir or settings.get("output_dir", "benchmark_results")
    
    # Print configuration
    if not args.quiet:
        print("\n" + "=" * 60)
        print("PIPELINE LLM BENCHMARK")
        print("=" * 60)
        print(f"Models to test: {len(models)}")
        for m in models:
            print(f"  • {m}")
        print(f"Tasks: {[t.value for t in tasks]}")
        print(f"Output: {output_dir}")
        print("=" * 60 + "\n")
    
    # Create and run benchmark
    benchmark = PipelineBenchmark(
        models=models,
        ground_truth_dir=settings.get("ground_truth_dir", "data/ground_truth"),
        ocr_cache_dir=settings.get("ocr_cache_dir", "evaluation_results_pipeline"),
        pdf_dir=settings.get("pdf_dir", "data/pdfs"),
        output_dir=output_dir,
    )
    
    # Prepare test data (run OCR on PDFs if needed)
    if not args.skip_prepare:
        if not args.quiet:
            print("Preparing test data...")
        prepared = benchmark.prepare_test_data()
        if not args.quiet:
            print(f"   {prepared} test cases ready\n")
    
    # If prepare-only mode, stop here
    if args.prepare_only:
        print("Test data preparation complete!")
        return
    
    try:
        report = benchmark.run(tasks=tasks)
        
        # Print summary
        print_benchmark_summary(report)
        
        print(f"\nResults saved to: {output_dir}/pipeline_benchmark_latest.json")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
