import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.benchmark_engine import run_benchmark


config = {
    "model_name": "gemini-2.0-flash",
    "provider": "google",
    "is_reasoning": False,
    "language": "EN",
    "sleep_seconds": 2.0,
    "max_questions": 500
}


if __name__ == "__main__":
    run_benchmark(config)