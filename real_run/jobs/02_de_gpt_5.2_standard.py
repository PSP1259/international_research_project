import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.benchmark_engine import run_benchmark



config = {
    "model_name": "gpt-5.2",
    "provider": "openai",
    "is_reasoning": False,    # Temperature 0.0
    "language": "DE",
    "sleep_seconds": 2.0,
    "max_questions": 600      
}



if __name__ == "__main__":
    run_benchmark(config)