import os
import requests
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Load .env file from project root (not ChatAgent directory)
# Go up from bot.py -> ChatAgent -> Agents -> src -> project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path)

# Also load from current directory as fallback
load_dotenv()

# API configuration
# bot.py ONLY uses the inference endpoint - it does NOT load the model
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:8000/message")

# Load tokenizer ONLY for building prompts (chat template formatting)
# The actual model inference is done via API endpoint
# Go up from bot.py -> ChatAgent -> Agents -> src -> project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-vl-8b-instruct")
MODEL_DIR = os.getenv("MODEL_DIR") or os.path.join(BASE_DIR, "models", MODEL_NAME)

# Ensure we use local files only - don't try to download from HuggingFace
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}. Please ensure models are in the models/ directory.")

# Load tokenizer only (NOT the model) - needed for chat template formatting
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

eos_token_id = tokenizer.eos_token_id
if isinstance(eos_token_id, list):
    stop_token_ids = eos_token_id
else:
    stop_token_ids = [eos_token_id] if eos_token_id is not None else []

def build_prompt(history, user_input):
    messages = [{"role": "system", "content": "You are a helpful chatbot."}]   
    for msg in history.messages:
        if msg.type == "human":
            messages.append({"role": "user", "content": msg.content})
        else:
            messages.append({"role": "assistant", "content": msg.content})   
    messages.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Disable reasoning output
    )
    return prompt


def pipe(prompt):
    """
    ONLY uses the inference API endpoint - does NOT load or use model directly.
    All model inference is handled by the API server (model.py).
    Returns response in the same format as the original pipeline.
    Uses the same parameters as the original pipeline configuration.
    """
    # Device-specific timeouts for 8B model
    # CUDA is fast (5-15s) → shorter timeout (60s) - fail fast if something wrong
    # MPS is slower (30-60s) → medium timeout (180s) - needs more time
    # CPU is slowest (2-5min) → longer timeout (300s) - needs most time
    
    # Try to detect device from Model API, fallback to environment variable
    device_timeout_map = {
        "cuda": 120,      # CUDA: Fast, 2 minutes should be plenty
        "mps": 300,       # MPS: Slower, 5 minutes for safety
        "cpu": 600,       # CPU: Slowest, 10 minutes for safety
    }
    
    # Check if device info is available from Model API
    device_type = None
    try:
        import requests as req_check
        device_check = req_check.get(MODEL_API_URL.replace("/message", "/"), timeout=5)
        if device_check.status_code == 200:
            device_info = device_check.json()
            device_str = device_info.get("device", "")
            if "cuda" in device_str.lower():
                device_type = "cuda"
            elif "mps" in device_str.lower():
                device_type = "mps"
            elif "cpu" in device_str.lower():
                device_type = "cpu"
    except:
        pass  # Fallback to environment variable
    
    # Use device-specific timeout or environment variable
    if device_type and device_type in device_timeout_map:
        default_timeout = device_timeout_map[device_type]
        print(f"[bot.py] Detected device: {device_type.upper()}, using timeout: {default_timeout}s")
    else:
        default_timeout = 300  # Safe default for unknown devices
    
    timeout_seconds = int(os.getenv("MODEL_API_TIMEOUT", str(default_timeout)))
    
    # Reduce max_new_tokens for 8B model to improve response time
    # 8B model is slower, so use smaller token limit by default
    default_max_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))  # Reduced from 512 for 8B model
    
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "prompt": prompt,
            "max_new_tokens": default_max_tokens,
            "do_sample": os.getenv("DO_SAMPLE", "False").lower() == "true",
            "return_full_text": False,
            "pad_token_id": tokenizer.pad_token_id
        }
        
        # Only add eos_token_id if we have valid stop tokens
        if stop_token_ids and len(stop_token_ids) > 0:
            # If single token, send as int; if multiple, send as list
            if len(stop_token_ids) == 1:
                data["eos_token_id"] = stop_token_ids[0]
            else:
                data["eos_token_id"] = stop_token_ids
        
        # Debug: Log request details
        print(f"[bot.py] Making request to: {MODEL_API_URL}")
        print(f"[bot.py] Prompt length: {len(prompt)}")
        print(f"[bot.py] Prompt preview: {prompt[:200]}...")
        print(f"[bot.py] Request timeout set to {timeout_seconds} seconds")
        print(f"[bot.py] Max new tokens: {default_max_tokens}")
        
        response = requests.post(MODEL_API_URL, json=data, headers=headers, timeout=timeout_seconds)
        
        # Check for errors before parsing JSON
        if response.status_code != 200:
            error_detail = response.text
            print(f"[bot.py] ERROR: Model API returned status {response.status_code}")
            print(f"[bot.py] Error response: {error_detail}")
            raise Exception(f"Model API error ({response.status_code}): {error_detail}")
        
        response.raise_for_status()
        
        result = response.json()
        generated_text = result.get("generated_text", "")
        
        print(f"[bot.py] Received response length: {len(generated_text)}")
        print(f"[bot.py] Response preview: {generated_text[:200]}...")
        
        # Return in the same format as the original pipeline
        return [{"generated_text": generated_text}]
    
    except requests.exceptions.Timeout as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[bot.py] ERROR: Model API request timed out after {timeout_seconds} seconds")
        print(f"[bot.py] This is common with the 8B model, especially on CPU/MPS")
        print(f"[bot.py] Suggestions:")
        print(f"  1. Reduce MAX_NEW_TOKENS (current: {default_max_tokens})")
        print(f"  2. Increase MODEL_API_TIMEOUT (current: {timeout_seconds}s)")
        print(f"  3. Use GPU (CUDA) for faster inference")
        print(f"[bot.py] Traceback:\n{error_trace}")
        raise Exception(f"Model API request timed out after {timeout_seconds} seconds. The 8B model is slower - try reducing max_new_tokens or using GPU.")
    except requests.exceptions.RequestException as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[bot.py] ERROR: API request failed: {str(e)}")
        print(f"[bot.py] Traceback:\n{error_trace}")
        raise Exception(f"API request failed: {str(e)}")

