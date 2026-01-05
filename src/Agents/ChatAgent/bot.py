import os
import requests
from dotenv import load_dotenv

# Load .env file from project root (not ChatAgent directory)
# Go up from bot.py -> ChatAgent -> Agents -> src -> project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path)

# Also load from current directory as fallback
load_dotenv()

# API configuration
# bot.py ONLY uses the Model API endpoint - it does NOT load models or tokenizers locally
# The Model API URL is configured via MODEL_API_URL environment variable
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://localhost:8000/message")

# Log the Model API URL being used (for debugging)
print(f"[bot.py] Model API URL configured: {MODEL_API_URL}")
print(f"[bot.py] Using Model API for all inference - bot.py does NOT load models or tokenizers")


def build_prompt(history, user_input):
    """
    Build a simple prompt format from conversation history.
    The Model API will handle proper chat template formatting.
    """
    messages = [{"role": "system", "content": "You are a helpful chatbot."}]   
    for msg in history.messages:
        if msg.type == "human":
            messages.append({"role": "user", "content": msg.content})
        else:
            messages.append({"role": "assistant", "content": msg.content})   
    messages.append({"role": "user", "content": user_input})

    # Build a simple conversational prompt format
    # The Model API should handle proper chat template formatting
    prompt_parts = []
    for msg in messages:
        if msg["role"] == "system":
            prompt_parts.append(f"System: {msg['content']}")
        elif msg["role"] == "user":
            prompt_parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            prompt_parts.append(f"Assistant: {msg['content']}")
    prompt = "\n".join(prompt_parts) + "\nAssistant:"
    return prompt


def pipe(prompt):
    """
    Sends prompt to Model API endpoint (configured via MODEL_API_URL environment variable).
    
    This function ONLY uses the inference API endpoint - it does NOT load or use model directly.
    All model inference is handled by the Model API server.
    
    The Model API URL is read from MODEL_API_URL environment variable (default: http://localhost:8000/message).
    
    Returns response in the same format as the original pipeline.
    """
    # Device-specific timeouts
    device_timeout_map = {
        "cuda": 120,      # CUDA: Fast, 2 minutes should be plenty
        "mps": 300,       # MPS: Slower, 5 minutes for safety
        "cpu": 600,       # CPU: Slowest, 10 minutes for safety
    }
    
    # Check if device info is available from Model API
    device_type = None
    try:
        device_check = requests.get(MODEL_API_URL.replace("/message", "/"), timeout=5)
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
    
    # Max new tokens configuration
    default_max_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))
    
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        # Build request payload - Model API will handle tokenization and formatting
        data = {
            "prompt": prompt,
            "max_new_tokens": default_max_tokens,
            "do_sample": os.getenv("DO_SAMPLE", "False").lower() == "true",
            "return_full_text": False
        }
        
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
        print(f"[bot.py] Suggestions:")
        print(f"  1. Reduce MAX_NEW_TOKENS (current: {default_max_tokens})")
        print(f"  2. Increase MODEL_API_TIMEOUT (current: {timeout_seconds}s)")
        print(f"  3. Use GPU (CUDA) for faster inference")
        print(f"[bot.py] Traceback:\n{error_trace}")
        raise Exception(f"Model API request timed out after {timeout_seconds} seconds. Try reducing max_new_tokens or using GPU.")
    except requests.exceptions.RequestException as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[bot.py] ERROR: API request failed: {str(e)}")
        print(f"[bot.py] Traceback:\n{error_trace}")
        raise Exception(f"API request failed: {str(e)}")
