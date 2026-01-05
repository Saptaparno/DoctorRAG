from typing import Optional, List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq
import torch
from contextlib import asynccontextmanager

# Model path
MODEL_PATH = "./models/qwen3-vl-8b-instruct"

# Global variables for model and tokenizer
model = None
tokenizer = None

# Store the last response
last_response = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown"""
    global model, tokenizer
    
    print("Loading Qwen3-VL-8B-Instruct model...")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        # Load model with automatic device detection (CUDA > MPS > CPU)
        # Note: Qwen3-VL-8B-Instruct is a vision-language model
        # Try AutoModelForVision2Seq first, fallback to AutoModelForCausalLM for text-only
        
        # Detect available device in priority order: CUDA > MPS > CPU
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
            print(f"✅ CUDA detected: Using GPU ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16  # MPS works better with float16
            print(f"✅ MPS detected: Using Apple Silicon GPU")
        else:
            device = "cpu"
            dtype = torch.float32
            print(f"⚠️  No GPU detected: Using CPU (slower performance)")
        
        # Try to load as vision-language model first, fallback to causal LM
        try:
            print("Attempting to load as vision-language model (AutoModelForVision2Seq)...")
            if device == "cuda":
                model = AutoModelForVision2Seq.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True
                )
                print(f"Model loaded successfully on CUDA with bfloat16 (Vision2Seq)")
            else:
                model = AutoModelForVision2Seq.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                model = model.to(device)
                print(f"Model loaded successfully on {device.upper()} with {dtype} (Vision2Seq)")
        except (ValueError, KeyError, AttributeError) as e:
            print(f"Vision2Seq loading failed: {e}")
            print("Falling back to AutoModelForCausalLM (text-only mode)...")
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True
                )
                print(f"Model loaded successfully on CUDA with bfloat16 (CausalLM)")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                model = model.to(device)
                print(f"Model loaded successfully on {device.upper()} with {dtype} (CausalLM)")
        
        model.eval()
        print("Model ready to accept requests")
        print(f"Model API will be available at http://0.0.0.0:8000")
        print(f"API Documentation: http://localhost:8000/docs")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")


app = FastAPI(
    title="Qwen3-VL-8B-Instruct Model API",
    description="API for Qwen3-VL-8B-Instruct vision-language model - POST to send messages, GET to receive responses",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint - API information."""
    # Detect current device
    if model is not None:
        try:
            device = next(model.parameters()).device
            device_type = str(device)
        except:
            device_type = "unknown"
    else:
        device_type = "not_loaded"
    
    return {
        "service": "Qwen3-VL-8B-Instruct Model API",
        "version": "1.0.0",
        "status": "running",
        "device": device_type,
        "endpoints": {
            "message": "POST /message - Send message to model",
            "response": "GET /response - Get last response",
            "docs": "GET /docs - Interactive API documentation",
            "openapi": "GET /openapi.json - OpenAPI schema"
        }
    }


class MessageRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20
    do_sample: Optional[bool] = True
    eos_token_id: Optional[Union[int, List[int]]] = None
    pad_token_id: Optional[int] = None
    return_full_text: Optional[bool] = False


class MessageResponse(BaseModel):
    generated_text: str


class ResponseData(BaseModel):
    generated_text: str


@app.get("/message")
async def message_info():
    """GET handler for /message - provides usage information"""
    return {
        "endpoint": "/message",
        "method": "POST",
        "description": "Send a message to the model for processing",
        "usage": {
            "method": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "prompt": "string (required)",
                "max_new_tokens": "int (optional, default: 512)",
                "temperature": "float (optional, default: 0.6)",
                "top_p": "float (optional, default: 0.95)",
                "top_k": "int (optional, default: 20)",
                "do_sample": "bool (optional, default: true)",
                "return_full_text": "bool (optional, default: false)"
            },
            "example": {
                "prompt": "Hello, how are you?",
                "max_new_tokens": 100
            }
        },
        "note": "This endpoint requires a POST request with JSON body. Use a tool like curl, Postman, or the /docs endpoint to test it."
    }


@app.post("/message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """
    Send a message to the model for processing.
    Returns the generated response directly.
    """
    global last_response
    
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please try again later."
        )
    
    try:
        # Debug: Log prompt (first 200 chars)
        prompt_preview = request.prompt[:200] + "..." if len(request.prompt) > 200 else request.prompt
        print(f"[model.py] Received prompt (length: {len(request.prompt)}): {prompt_preview}")
        
        # Format prompt using chat template if available (Qwen3-VL models expect chat format)
        # If prompt already looks formatted (contains special tokens), use as-is
        # Otherwise, format it as a simple user message
        if "<|im_start|>" in request.prompt or "<|im_end|>" in request.prompt:
            # Prompt is already formatted
            formatted_prompt = request.prompt
        else:
            # Format as a simple chat message
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": request.prompt}
                ]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                print(f"[model.py] Formatted prompt using chat template")
                print(f"[model.py] Formatted prompt preview: {formatted_prompt[:300]}...")
            except Exception as e:
                # Fallback to raw prompt if chat template fails
                print(f"[model.py] Warning: Could not apply chat template: {e}. Using raw prompt.")
                formatted_prompt = request.prompt
        
        # Tokenize input
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move inputs to the same device as the model
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Prepare generation parameters
        eos_token_id = request.eos_token_id
        if eos_token_id is None:
            try:
                eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            except Exception as e:
                print(f"[model.py] Warning: Could not get tokenizer.eos_token_id: {e}")
                eos_token_id = None
        
        pad_token_id = request.pad_token_id
        if pad_token_id is None:
            try:
                pad_token_id = getattr(tokenizer, 'pad_token_id', None)
                if pad_token_id is None:
                    pad_token_id = getattr(tokenizer, 'eos_token_id', None)
            except Exception as e:
                print(f"[model.py] Warning: Could not get pad_token_id: {e}")
                pad_token_id = None
        
        # Generate
        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": request.max_new_tokens,
                "do_sample": request.do_sample,
            }
            
            # Add repetition_penalty if supported (most models support this)
            try:
                generate_kwargs["repetition_penalty"] = 1.1  # Prevent repetition
            except Exception:
                pass  # Some models might not support this parameter
            
            # Add pad_token_id if available
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            
            # Add eos_token_id if available (can be int, list, or None)
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = eos_token_id
            
            # Only add sampling parameters if do_sample is True
            if request.do_sample:
                if request.temperature is not None:
                    generate_kwargs["temperature"] = request.temperature
                if request.top_p is not None:
                    generate_kwargs["top_p"] = request.top_p
                if request.top_k is not None:
                    generate_kwargs["top_k"] = request.top_k
            
            # Debug: Log generation parameters
            print(f"[model.py] Generation parameters:")
            print(f"  - max_new_tokens: {generate_kwargs.get('max_new_tokens')}")
            print(f"  - do_sample: {generate_kwargs.get('do_sample')}")
            print(f"  - eos_token_id: {generate_kwargs.get('eos_token_id')}")
            print(f"  - pad_token_id: {generate_kwargs.get('pad_token_id')}")
            print(f"  - repetition_penalty: {generate_kwargs.get('repetition_penalty')}")
            print(f"  - Input shape: {inputs['input_ids'].shape}")
            
            try:
                outputs = model.generate(**generate_kwargs)
            except Exception as gen_error:
                print(f"[model.py] ERROR in model.generate(): {gen_error}")
                import traceback
                print(traceback.format_exc())
                raise
        
        # Decode output
        if request.return_full_text:
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Extract only the newly generated tokens
            input_length = inputs['input_ids'].shape[1]
            generated_text = tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
        
        # Debug: Log generated text (first 200 chars)
        generated_preview = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
        print(f"[model.py] Generated text (length: {len(generated_text)}): {generated_preview}")
        
        # Clean up the generated text - remove any continuation artifacts
        # Stop at special tokens first
        if "<|im_end|>" in generated_text:
            generated_text = generated_text.split("<|im_end|>")[0].strip()
        if "<|endoftext|>" in generated_text:
            generated_text = generated_text.split("<|endoftext|>")[0].strip()
        
        # Stop if model starts generating user/assistant tags (continuing conversation incorrectly)
        if "\n<|im_start|>user" in generated_text:
            generated_text = generated_text.split("\n<|im_start|>user")[0].strip()
        if "\n<|im_start|>assistant" in generated_text:
            generated_text = generated_text.split("\n<|im_start|>assistant")[0].strip()
        
        # If text seems to be rambling (very long, multiple paragraphs), try to find a natural stop
        if len(generated_text) > 300:
            # Try to find sentence boundaries
            sentences = generated_text.split('. ')
            if len(sentences) > 3:
                # Take first few complete sentences
                reasonable_reply = '. '.join(sentences[:3]) + '.'
                if len(reasonable_reply) > 30:  # Only use if it's substantial
                    generated_text = reasonable_reply.strip()
        
        # Final cleanup - remove any trailing incomplete sentences
        if generated_text.count('.') > 0:
            # Keep only complete sentences
            parts = generated_text.split('.')
            if len(parts) > 1:
                # Take all but the last part (which might be incomplete) and add periods back
                complete_sentences = '. '.join(parts[:-1]) + '.'
                if len(complete_sentences) > 20:
                    generated_text = complete_sentences.strip()
        
        # If the original prompt was very short (like "Hi"), the response should also be short
        # If we got a very long response to a short prompt, it's likely rambling
        original_prompt_length = len(request.prompt.strip())
        if original_prompt_length < 10 and len(generated_text) > 100:
            # For very short prompts, take only the first sentence or first 50 chars
            first_sentence = generated_text.split('.')[0] + '.' if '.' in generated_text else generated_text
            if len(first_sentence) <= 100:
                generated_text = first_sentence.strip()
            else:
                # Just take first 50 chars for very short prompts
                generated_text = generated_text[:50].strip()
                if not generated_text.endswith(('.', '!', '?')):
                    generated_text += '.'
            print(f"[model.py] Truncated long response ({len(generated_text)} chars) for short prompt ({original_prompt_length} chars)")
        
        # Store last response
        last_response = {
            "generated_text": generated_text
        }
        
        return MessageResponse(
            generated_text=generated_text
        )
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[model.py] ERROR during inference: {str(e)}")
        print(f"[model.py] Traceback:\n{error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during inference: {str(e)}"
        )


@app.get("/response", response_model=ResponseData)
async def get_response():
    """
    Retrieve the last response from the model.
    """
    if last_response is None:
        raise HTTPException(
            status_code=404,
            detail="No response available. Send a message first using POST /message"
        )
    
    return ResponseData(
        generated_text=last_response["generated_text"]
    )


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting Model API Server...")
    print("=" * 60)
    print(f"Host: 0.0.0.0")
    print(f"Port: 8000")
    print("=" * 60)
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        print(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        raise

