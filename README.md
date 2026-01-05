# DoctorRAG

A Retrieval-Augmented Generation (RAG) system for medical appointment scheduling and triage, powered by Qwen3-VL-8B-Instruct and LangGraph workflow orchestration.

## 🎯 Overview

DoctorRAG is an intelligent medical assistant that helps users:
- **Triage medical symptoms** and assess priority
- **Match patients with appropriate providers** based on symptoms
- **Schedule appointments** using semantic search over available slots
- **Book appointments** with human-in-the-loop confirmation
- **Chat naturally** with a medical AI assistant

The system uses a multi-agent architecture orchestrated by LangGraph, with each agent specialized in a specific task (triage, provider matching, scheduling, booking).

## ✨ Features

- 🤖 **Multi-Agent Architecture**: Specialized agents for triage, provider matching, scheduling, and booking
- 🧠 **RAG-Powered**: Uses semantic search with HuggingFace embeddings for intelligent slot matching
- 💬 **Natural Language Interface**: Chat-based interaction with automatic intent detection
- 🔄 **Workflow Orchestration**: LangGraph-based workflow that routes requests through appropriate agents
- 🎯 **Intent Detection**: Automatically detects medical requests and scheduling needs
- 📅 **Smart Scheduling**: Semantic search finds the best appointment slots based on user requirements
- ✅ **Human-in-the-Loop**: Booking confirmation with patient information collection
- 🚀 **GPU Acceleration**: Supports CUDA, MPS (Apple Silicon), and CPU inference
- 📝 **Session Management**: Maintains conversation history per session

## 🏗️ Architecture

```
┌─────────────┐
│   User      │
│  (Chat)     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│      ChatAgent (FastAPI)             │
│  - Intent Detection                  │
│  - Conversation Management           │
│  - Workflow Triggering               │
│                                      │
│  ┌──────────────────────────────┐   │
│  │  Uses Model API via HTTP     │   │
│  │  (MODEL_API_URL env var)     │   │
│  └──────────────────────────────┘   │
└──────┬───────────────────────────────┘
       │ HTTP Request (MODEL_API_URL)
       │
       ▼
┌─────────────────────────────────────┐
│   Model API (FastAPI)                │
│   Port: 8000                         │
│   Qwen3-VL-8B-Instruct              │
│   - CUDA / MPS / CPU                 │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│      LangGraph Workflow              │
│  ┌──────────┐  ┌──────────┐         │
│  │  Triage  │→ │ Provider │         │
│  │  Agent   │  │ Matching │         │
│  └──────────┘  └────┬─────┘         │
│                     │                │
│              ┌──────▼─────┐          │
│              │ Scheduling │          │
│              │   Agent    │          │
│              └──────┬──────┘          │
│                     │                │
│              ┌──────▼──────┐         │
│              │   Booking   │         │
│              │   Agent     │         │
│              └─────────────┘         │
└─────────────────────────────────────┘
```

**Key Architecture Points**:
- **ChatAgent** makes HTTP requests to **Model API** for all LLM inference
- Connection URL is configured via `MODEL_API_URL` environment variable
- Model API runs as a separate service (port 8000)
- ChatAgent does NOT load the model - it's a client to the Model API

### Components

1. **ChatAgent** (`src/Agents/ChatAgent/`): Main entry point, handles user interactions
   - **Uses Model API**: ChatAgent connects to the Model API (via `MODEL_API_URL` env var) for LLM inference
   - Does NOT load the model directly - all inference is done through the Model API
2. **TriageAgent** (`src/Agents/TriageAgent/`): Assesses symptoms and determines priority
3. **ProviderMatchingAgent** (`src/Agents/ProviderMatchingAgent/`): Matches patients with appropriate providers
4. **SchedulingAgent** (`src/Agents/SchedulingAgent/`): Finds available appointment slots using RAG
5. **BookingAgent** (`src/Agents/BookingAgent/`): Handles appointment booking and confirmation
6. **Model API** (`model.py`): Qwen3-VL-8B-Instruct inference service
   - **Standalone service**: Runs independently on port 8000
   - **Used by ChatAgent**: ChatAgent makes HTTP requests to this service
7. **Workflow** (`src/workflow.py`): LangGraph orchestration of all agents

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)
- Apple Silicon Mac (optional, for MPS acceleration)
- ~20GB disk space for model files
- 8GB+ RAM (16GB+ recommended)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd DoctorRAG
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Model

The system uses **Qwen3-VL-8B-Instruct**. You need to download it manually:

```bash
# Option 1: Using HuggingFace CLI
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir ./models/qwen3-vl-8b-instruct

# Option 2: Using Python
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='./models/qwen3-vl-8b-instruct')"
```

**Note**: The model is ~16GB. Ensure you have sufficient disk space and internet bandwidth.

### 4. Verify Model Installation

```bash
ls -lh ./models/qwen3-vl-8b-instruct/
```

You should see model files including:
- `config.json`
- `tokenizer.json`
- `model.safetensors` (or similar)

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```env
# Model API Configuration
# ChatAgent uses this URL to connect to the Model API service
MODEL_API_URL=http://localhost:8000/message
MODEL_NAME=qwen3-vl-8b-instruct
MODEL_DIR=./models/qwen3-vl-8b-instruct

# ChatAgent Configuration
CHATAGENT_API_URL=http://localhost:8001

# Model Inference Settings
MAX_NEW_TOKENS=256
DO_SAMPLE=False
MODEL_API_TIMEOUT=300
MODEL_API_TIMEOUT_CUDA=120
MODEL_API_TIMEOUT_MPS=300
MODEL_API_TIMEOUT_CPU=600
```

**Important**: 
- `MODEL_API_URL` is used by **ChatAgent** to connect to the **Model API** service
- ChatAgent does NOT load the model directly - it makes HTTP requests to the Model API
- Both services must be running simultaneously for the system to work

### Default Ports

- **Model API**: `8000`
- **ChatAgent API**: `8001`

## 🎮 Usage

### Starting the Services

You need to run **two services** simultaneously. **ChatAgent depends on the Model API**, so start the Model API first:

#### Terminal 1: Model API (Start First)

```bash
python model.py
```

The model will load automatically. You'll see:
```
✅ CUDA detected: Using GPU (...)
# or
✅ MPS detected: Using Apple Silicon GPU
# or
⚠️  No GPU detected: Using CPU (slower performance)
Model ready to accept requests
Model API will be available at http://0.0.0.0:8000
```

**Important**: Wait for "Model ready to accept requests" before starting ChatAgent.

#### Terminal 2: ChatAgent API (Start After Model API)

```bash
cd src/Agents/ChatAgent
python ChatAgent.py
```

Or from project root:
```bash
python -m src.Agents.ChatAgent.ChatAgent
```

**Connection**: ChatAgent will connect to the Model API using the `MODEL_API_URL` environment variable (default: `http://localhost:8000/message`). If the Model API is not running, ChatAgent will fail to process chat messages.

### API Endpoints

#### ChatAgent API (Port 8001)

**Main Chat Endpoint**
```bash
POST http://localhost:8001/chat
Content-Type: application/json

{
  "message": "I am having chest pain",
  "session_id": "user123",
  "context": {},
  "patient_info": {}
}
```

**Response**
```json
{
  "reply": "I understand you're experiencing chest pain. This is a serious symptom...",
  "session_id": "user123",
  "workflow_triggered": true
}
```

**Clear Session**
```bash
POST http://localhost:8001/session/clear
Content-Type: application/json

{
  "session_id": "user123"
}
```

**Booking Confirmation**
```bash
POST http://localhost:8001/booking/confirm
Content-Type: application/json

{
  "slot_id": "slot_123",
  "patient_name": "John Doe",
  "patient_contact": "john@example.com",
  "appointment_details": {
    "provider_type": "cardiologist",
    "provider_name": "Dr. Smith",
    "date": "2024-01-15",
    "time": "10:00 AM",
    "duration_minutes": 30
  }
}
```

#### Model API (Port 8000)

**Send Message**
```bash
POST http://localhost:8000/message
Content-Type: application/json

{
  "prompt": "Hello, how can you help me?",
  "max_new_tokens": 256,
  "do_sample": false,
  "temperature": 0.6
}
```

**Response**
```json
{
  "generated_text": "I'm here to help you with medical appointments..."
}
```

**Get Last Response**
```bash
GET http://localhost:8000/response
```

### Interactive API Documentation

Both services provide interactive Swagger UI:

- **ChatAgent**: http://localhost:8001/docs
- **Model API**: http://localhost:8000/docs

### Example Workflow

1. **User sends message**: "I am having chest pain"
2. **ChatAgent detects medical intent** → Triggers workflow
3. **Triage Agent** assesses symptoms → Determines priority (urgent/high/medium/low)
4. **Provider Matching Agent** matches to appropriate provider (cardiologist)
5. **Scheduling Agent** finds available slots using RAG
6. **User confirms booking** → Booking Agent creates appointment

## 📁 Project Structure

```
DoctorRAG/
├── model.py                          # Model API service
├── requirements.txt                  # Python dependencies
├── .env                             # Environment variables (create this)
├── models/                          # Model files directory
│   └── qwen3-vl-8b-instruct/       # Qwen3-VL-8B-Instruct model
├── src/
│   ├── workflow.py                  # LangGraph workflow orchestration
│   ├── knowledge_base.py            # RAG knowledge base setup
│   └── Agents/
│       ├── ChatAgent/
│       │   ├── ChatAgent.py        # Main FastAPI application
│       │   ├── chat.py             # Chat flow logic
│       │   ├── bot.py              # Model API client
│       │   └── history.py          # Session history management
│       ├── TriageAgent/
│       │   └── Triage.py           # Triage logic
│       ├── ProviderMatchingAgent/
│       │   └── ProviderMatching.py # Provider matching logic
│       ├── SchedulingAgent/
│       │   └── Scheduling.py       # RAG-based scheduling
│       └── BookingAgent/
│           └── Booking.py          # Booking logic
```

## 🔧 Troubleshooting

### Model Not Loading

**Error**: `KeyError: 'qwen3_vl'` or `Model type not recognized`

**Solution**: Update transformers:
```bash
pip install --upgrade transformers>=4.52.0
# or
pip install git+https://github.com/huggingface/transformers.git
```

### Timeout Errors

**Error**: `Read timed out. (read timeout=120)`

**Solution**: The 8B model is slower on CPU/MPS. Increase timeout:
```bash
export MODEL_API_TIMEOUT=600  # 10 minutes for CPU
export MODEL_API_TIMEOUT_MPS=300  # 5 minutes for MPS
```

### Out of Memory

**Error**: `CUDA out of memory` or `MPS out of memory`

**Solution**: 
- Reduce `MAX_NEW_TOKENS` in `.env`
- Use CPU instead (slower but more memory)
- Use quantization (4-bit/8-bit) if available

### Port Already in Use

**Error**: `Address already in use`

**Solution**: 
- Check if services are already running: `lsof -i :8000` or `lsof -i :8001`
- Kill existing processes or change ports in code

### Model API Not Responding

**Error**: `Connection refused` or `500 Internal Server Error` or `API request failed`

**Solution**:
1. **Check Model API is running**: `curl http://localhost:8000/`
2. **Verify MODEL_API_URL**: Ensure ChatAgent is using the correct URL
   ```bash
   # Check environment variable
   echo $MODEL_API_URL
   # Should be: http://localhost:8000/message
   ```
3. **Check Model API logs**: Look for errors in the Model API terminal
4. **Verify model files exist**: `ls ./models/qwen3-vl-8b-instruct/`
5. **Start Model API first**: ChatAgent requires Model API to be running before it starts

### ChatAgent Cannot Connect to Model API

**Error**: `API request failed: Connection refused` or `API request timed out`

**Solution**:
1. **Ensure Model API is running**: Start `model.py` first, wait for "Model ready to accept requests"
2. **Check MODEL_API_URL**: Verify the environment variable points to the correct Model API URL
   ```bash
   # In .env file or environment
   MODEL_API_URL=http://localhost:8000/message
   ```
3. **Verify port 8000 is accessible**: `curl http://localhost:8000/`
4. **Check firewall/network**: Ensure localhost connections are allowed
5. **Restart both services**: Stop both, start Model API first, then ChatAgent

## 🎯 Intent Detection

The system automatically detects:

- **Medical Intent**: Keywords like "pain", "symptom", "hurt", "ache"
- **Scheduling Intent**: Keywords like "schedule", "book", "appointment", "see a doctor"

When detected, the workflow is automatically triggered.

## 🔐 Security Notes

- **No API Key Authentication**: The current implementation does not require API keys
- **Local Development**: Services run on `localhost` by default
- **Production**: Add authentication, rate limiting, and HTTPS before deploying

## 📊 Performance

### Inference Speed (256 tokens)

| Device | Time | Tokens/sec |
|--------|------|------------|
| CUDA (RTX 4090) | 5-15s | ~17-51 |
| CUDA (RTX 4060) | 15-30s | ~8-17 |
| MPS (M4 Pro) | 20-40s | ~6-13 |
| CPU | 60-120s | ~2-4 |

### Memory Requirements

- **Model Size**: ~8GB (float16) or ~16GB (float32)
- **VRAM/RAM**: 8GB minimum, 16GB+ recommended
- **Disk Space**: ~20GB for model files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- **Qwen3-VL-8B-Instruct**: Model by Alibaba Cloud
- **LangGraph**: Workflow orchestration
- **FastAPI**: Web framework
- **HuggingFace**: Transformers and embeddings

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the API documentation at `/docs` endpoints

---

**Note**: This is a development/demo system. For production use, add proper authentication, error handling, logging, and security measures.

