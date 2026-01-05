from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from chat import chatflow
from history import clear_session
from datetime import datetime
import uuid
import sys
import os

# Add Agents directory to path to import agent functions
agents_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if agents_dir not in sys.path:
    sys.path.insert(0, agents_dir)

from Agents.TriageAgent.Triage import triage
from Agents.ProviderMatchingAgent.ProviderMatching import match_provider
from Agents.SchedulingAgent.Scheduling import schedule_appointment

app = FastAPI(
    title="DoctorRAG Chat Agent",
    description="Chat agent API that uses the model inference endpoints with human-in-the-loop booking",
    version="1.0.0"
)


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "service": "DoctorRAG Chat Agent",
        "version": "1.0.0",
        "status": "running",
        "public_endpoints": {
            "chat": "/chat - Main chat endpoint (triggers workflow automatically)",
            "booking_confirm": "/booking/confirm - Confirm and book appointment",
            "session_clear": "/session/clear - Clear conversation history",
            "docs": "/docs - Interactive API documentation"
        },
        "note": "The /chat endpoint automatically detects medical/scheduling intent and triggers the workflow internally. Other endpoints are for internal use."
    }


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    context: Optional[Dict[str, Any]] = None
    patient_info: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    workflow_triggered: Optional[bool] = False


class SessionClearRequest(BaseModel):
    session_id: Optional[str] = "default"


class SessionClearResponse(BaseModel):
    message: str
    session_id: str


class BookingConfirmationRequest(BaseModel):
    slot_id: str
    patient_name: str
    patient_contact: str
    appointment_details: Dict[str, Any]
    additional_info: Optional[Dict[str, Any]] = None


class BookingConfirmationResponse(BaseModel):
    booking_id: str
    confirmation_code: str
    status: str
    appointment: Dict[str, Any]
    patient: Dict[str, Any]
    booking_time: str
    message: str


class TriageRequest(BaseModel):
    symptoms: str
    context: Optional[Dict[str, Any]] = None


class TriageResponse(BaseModel):
    priority: str
    assessment: str
    recommended_action: str
    symptoms: str
    context: Optional[Dict[str, Any]] = None


class ProviderMatchingRequest(BaseModel):
    symptoms: str
    priority: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ProviderMatchingResponse(BaseModel):
    matched_providers: list
    primary_provider: Dict[str, Any]
    reasoning: str
    symptoms: str
    priority: Optional[str] = None


class SchedulingRequest(BaseModel):
    request: str
    context: Optional[Dict[str, Any]] = None


class SchedulingResponse(BaseModel):
    available_slots: list
    recommended_slot: Dict[str, Any]
    reasoning: str
    request: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the chat agent and receive a response.
    The agent maintains conversation history per session_id.
    If medical request intent is detected, triggers workflow from triage.
    If scheduling intent is detected, triggers workflow from chat node.
    Otherwise, uses normal chatbot conversation.
    """
    try:
        # Check if scheduling or medical request intent is detected
        scheduling_keywords = [
            "schedule", "book", "appointment", "see a doctor", "make an appointment",
            "need to see", "want to see", "book me", "schedule me"
        ]
        medical_keywords = [
            "pain", "hurt", "ache", "symptom", "feeling", "unwell", "sick", "ill",
            "fever", "cough", "headache", "nausea", "dizzy", "chest pain", "stomach",
            "rash", "bleeding", "injury", "wound", "infection", "problem", "issue"
        ]
        user_lower = request.message.lower()
        workflow_triggered = any(keyword in user_lower for keyword in scheduling_keywords) or \
                            any(keyword in user_lower for keyword in medical_keywords)
        
        reply = chatflow(
            user_text=request.message,
            session_id=request.session_id,
            context=request.context,
            patient_info=request.patient_info
        )
        
        return ChatResponse(
            reply=reply,
            session_id=request.session_id,
            workflow_triggered=workflow_triggered
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ChatAgent] ERROR processing chat message: {str(e)}")
        print(f"[ChatAgent] Traceback:\n{error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat message: {str(e)}"
        )


@app.post("/session/clear", response_model=SessionClearResponse)
async def clear_session_endpoint(request: SessionClearRequest):
    """
    Clear the conversation history for a specific session.
    """
    try:
        clear_session(request.session_id)
        return SessionClearResponse(
            message=f"Session '{request.session_id}' cleared successfully",
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing session: {str(e)}"
        )


@app.post("/triage", response_model=TriageResponse, include_in_schema=False)
async def triage_endpoint(request: TriageRequest):
    """
    Triage agent endpoint - INTERNAL USE ONLY.
    Called by workflow internally. Users should use /chat endpoint.
    """
    try:
        result = triage(request.symptoms, request.context)
        return TriageResponse(
            priority=result.get("priority", "unknown"),
            assessment=result.get("assessment", ""),
            recommended_action=result.get("recommended_action", ""),
            symptoms=result.get("symptoms", request.symptoms),
            context=result.get("context")
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in triage: {str(e)}"
        )


@app.post("/provider/matching", response_model=ProviderMatchingResponse, include_in_schema=False)
async def provider_matching_endpoint(request: ProviderMatchingRequest):
    """
    Provider matching agent endpoint - INTERNAL USE ONLY.
    Called by workflow internally. Users should use /chat endpoint.
    """
    try:
        result = match_provider(
            symptoms=request.symptoms,
            priority=request.priority,
            context=request.context
        )
        return ProviderMatchingResponse(
            matched_providers=result.get("matched_providers", []),
            primary_provider=result.get("primary_provider", {}),
            reasoning=result.get("reasoning", ""),
            symptoms=result.get("symptoms", request.symptoms),
            priority=result.get("priority")
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in provider matching: {str(e)}"
        )


@app.post("/scheduling", response_model=SchedulingResponse, include_in_schema=False)
async def scheduling_endpoint(request: SchedulingRequest):
    """
    Scheduling agent endpoint - INTERNAL USE ONLY.
    Called by workflow internally. Users should use /chat endpoint.
    """
    try:
        # Extract provider type and priority from context if available
        context = request.context or {}
        provider_type = context.get("provider_type")
        priority = context.get("priority")
        
        # Build context with provider type and priority
        scheduling_context = {
            **context,
            "provider_type": provider_type,
            "priority": priority
        }
        
        result = schedule_appointment(
            request=request.request,
            context=scheduling_context
        )
        return SchedulingResponse(
            available_slots=result.get("available_slots", []),
            recommended_slot=result.get("recommended_slot", {}),
            reasoning=result.get("reasoning", ""),
            request=result.get("request", request.request)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in scheduling: {str(e)}"
        )


# In-memory booking storage (in production, this would be a database)
_bookings = {}


@app.post("/booking/confirm", response_model=BookingConfirmationResponse)
async def confirm_booking(request: BookingConfirmationRequest):
    """
    Human-in-the-loop booking confirmation endpoint.
    After ChatAgent suggests an appointment, user confirms and this endpoint books it.
    Pure booking logic (no RAG) - creates the actual booking.
    """
    try:
        # Generate unique booking ID
        booking_id = str(uuid.uuid4())
        
        # Generate confirmation code (6-digit code)
        confirmation_code = str(uuid.uuid4().int)[:6].zfill(6)
        
        # Prepare patient info
        patient_info = {
            "name": request.patient_name,
            "contact": request.patient_contact
        }
        
        if request.additional_info:
            patient_info.update(request.additional_info)
        
        # Create booking record
        booking = {
            "booking_id": booking_id,
            "slot_id": request.slot_id,
            "status": "confirmed",
            "patient_info": patient_info,
            "appointment_details": request.appointment_details,
            "booking_time": datetime.now().isoformat(),
            "confirmation_code": confirmation_code
        }
        
        # Store booking
        _bookings[booking_id] = booking
        
        # Prepare response
        booking_result = {
            "booking_id": booking_id,
            "confirmation_code": confirmation_code,
            "status": "confirmed",
            "appointment": {
                "provider_name": request.appointment_details.get("provider_name"),
                "provider_type": request.appointment_details.get("provider_type"),
                "date": request.appointment_details.get("date"),
                "time": request.appointment_details.get("time"),
                "duration_minutes": request.appointment_details.get("duration_minutes")
            },
            "patient": {
                "name": patient_info["name"],
                "contact": patient_info["contact"]
            },
            "booking_time": booking["booking_time"],
            "message": f"Appointment booked successfully. Confirmation code: {confirmation_code}"
        }
        
        return BookingConfirmationResponse(
            booking_id=booking_result["booking_id"],
            confirmation_code=booking_result["confirmation_code"],
            status=booking_result["status"],
            appointment=booking_result["appointment"],
            patient=booking_result["patient"],
            booking_time=booking_result["booking_time"],
            message=booking_result["message"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error confirming booking: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

