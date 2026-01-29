
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not Set')}")

try:
    from llama_cpp import Llama
    print("llama-cpp-python imported successfully")
except ImportError:
    print("Failed to import llama-cpp-python")
    sys.exit(1)

model_path = "models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    sys.exit(1)

print("Attempting to load model with n_gpu_layers=-1...")
try:
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        verbose=True
    )
    print(f"Model loaded. n_gpu_layers used: {llm.n_gpu_layers}")
    # Inspect internal state if possible
    print(f"Llama metadata: {llm.metadata}")
except Exception as e:
    print(f"Error loading model: {e}")
