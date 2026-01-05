"""
LangGraph workflow that wires up all agents from chat to booking.
Flow: Chat → Triage → Provider Matching → Scheduling → Human Confirmation → Booking
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
import sys
import os
import requests

# Import agent functions directly (more efficient than HTTP calls)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Agents"))

from Agents.TriageAgent.Triage import triage
from Agents.ProviderMatchingAgent.ProviderMatching import match_provider
from Agents.SchedulingAgent.Scheduling import schedule_appointment

# ChatAgent API configuration - only for booking endpoint
CHATAGENT_API_URL = os.getenv("CHATAGENT_API_URL", "http://localhost:8001")


class WorkflowState(TypedDict):
    """State for the workflow."""
    # User input
    user_message: str
    session_id: str
    context: dict
    
    # Triage results
    triage_result: dict
    priority: str
    symptoms: str
    
    # Provider matching results
    provider_match_result: dict
    matched_provider: dict
    
    # Scheduling results
    scheduling_result: dict
    recommended_slot: dict
    available_slots: list
    
    # Booking results
    booking_confirmed: bool
    booking_result: dict
    patient_info: dict
    
    # Flow control
    next_step: str
    error: str


def chat_node(state: WorkflowState) -> WorkflowState:
    """
    Chat node - processes user message and determines if scheduling is needed.
    In a real implementation, this would call ChatAgent.
    For now, it extracts scheduling intent from the message.
    """
    user_message = state.get("user_message", "").lower()
    
    # Detect if user wants to schedule/book an appointment
    scheduling_keywords = ["schedule", "book", "appointment", "see a doctor", "make an appointment", "need to see"]
    needs_scheduling = any(keyword in user_message for keyword in scheduling_keywords)
    
    # Extract symptoms from message
    symptoms = state.get("user_message", "")
    
    if needs_scheduling:
        return {
            **state,
            "symptoms": symptoms,
            "next_step": "triage"
        }
    else:
        return {
            **state,
            "next_step": "end",
            "error": "No scheduling intent detected. Please ask about booking an appointment."
        }


def triage_node(state: WorkflowState) -> WorkflowState:
    """
    Triage node - assesses symptoms and determines priority.
    Calls agent function directly (not via HTTP).
    """
    try:
        symptoms = state.get("symptoms", state.get("user_message", ""))
        context = state.get("context", {})
        
        # Call triage function directly
        triage_result = triage(symptoms, context)
        
        return {
            **state,
            "triage_result": triage_result,
            "priority": triage_result.get("priority", "unknown"),
            "symptoms": triage_result.get("symptoms", symptoms),
            "next_step": "provider_matching"
        }
    except Exception as e:
        return {
            **state,
            "next_step": "end",
            "error": f"Triage error: {str(e)}"
        }


def provider_matching_node(state: WorkflowState) -> WorkflowState:
    """
    Provider matching node - matches patient with appropriate provider.
    Calls agent function directly (not via HTTP).
    """
    try:
        symptoms = state.get("symptoms", state.get("user_message", ""))
        priority = state.get("priority", "unknown")
        context = state.get("context", {})
        
        # Call provider matching function directly
        provider_match_result = match_provider(
            symptoms=symptoms,
            priority=priority,
            context=context
        )
        
        return {
            **state,
            "provider_match_result": provider_match_result,
            "matched_provider": provider_match_result.get("primary_provider", {}),
            "next_step": "scheduling"
        }
    except Exception as e:
        return {
            **state,
            "next_step": "end",
            "error": f"Provider matching error: {str(e)}"
        }


def scheduling_node(state: WorkflowState) -> WorkflowState:
    """
    Scheduling node - finds available appointment slots.
    Calls agent function directly (not via HTTP).
    """
    try:
        request = state.get("user_message", "")
        context = state.get("context", {})
        matched_provider = state.get("matched_provider", {})
        priority = state.get("priority", "unknown")
        
        # Add provider type and priority to context for scheduling
        scheduling_context = {
            **context,
            "provider_type": matched_provider.get("type"),
            "priority": priority
        }
        
        # Call scheduling function directly
        scheduling_result = schedule_appointment(
            request=request,
            context=scheduling_context
        )
        
        return {
            **state,
            "scheduling_result": scheduling_result,
            "recommended_slot": scheduling_result.get("recommended_slot", {}),
            "available_slots": scheduling_result.get("available_slots", []),
            "next_step": "human_confirmation"
        }
    except Exception as e:
        return {
            **state,
            "next_step": "end",
            "error": f"Scheduling error: {str(e)}"
        }


def human_confirmation_node(state: WorkflowState) -> WorkflowState:
    """
    Human confirmation node - waits for user confirmation before booking.
    This is a checkpoint where the system presents the recommended slot
    and waits for user confirmation.
    """
    recommended_slot = state.get("recommended_slot", {})
    available_slots = state.get("available_slots", [])
    
    # Check if recommended_slot is empty or missing required fields
    if not recommended_slot or not recommended_slot.get("slot_id"):
        return {
            **state,
            "next_step": "end",
            "error": "No recommended slot available. Please try again or provide more details."
        }
    
    # In a real implementation, this would wait for user confirmation
    # For now, we'll set booking_confirmed based on context or default to False
    # The actual confirmation should come from the ChatAgent API endpoint
    booking_confirmed = state.get("context", {}).get("confirm_booking", False)
    
    if booking_confirmed:
        return {
            **state,
            "next_step": "booking"
        }
    else:
        # Return state with recommendation for user to confirm
        return {
            **state,
            "next_step": "end",
            "booking_confirmed": False
        }


def booking_node(state: WorkflowState) -> WorkflowState:
    """
    Booking node - creates the actual booking via ChatAgent's /booking/confirm endpoint.
    """
    try:
        recommended_slot = state.get("recommended_slot", {})
        patient_info = state.get("patient_info", {})
        context = state.get("context", {})
        
        if not recommended_slot or not recommended_slot.get("slot_id"):
            return {
                **state,
                "next_step": "end",
                "error": "No valid slot to book"
            }
        
        # Extract patient information
        patient_name = patient_info.get("name") or context.get("patient_name", "Unknown")
        patient_contact = patient_info.get("contact") or context.get("patient_contact", "")
        
        if not patient_contact:
            return {
                **state,
                "next_step": "end",
                "error": "Patient contact information required for booking"
            }
        
        # Prepare appointment details
        appointment_details = {
            "provider_type": recommended_slot.get("provider_type"),
            "provider_name": recommended_slot.get("provider_name"),
            "date": recommended_slot.get("date"),
            "time": recommended_slot.get("time"),
            "duration_minutes": recommended_slot.get("duration_minutes")
        }
        
        # Create booking via ChatAgent's booking confirmation endpoint
        try:
            url = f"{CHATAGENT_API_URL}/booking/confirm"
            payload = {
                "slot_id": recommended_slot.get("slot_id"),
                "patient_name": patient_name,
                "patient_contact": patient_contact,
                "appointment_details": appointment_details,
                "additional_info": patient_info
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            booking_result = response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to confirm booking via ChatAgent: {str(e)}")
        
        return {
            **state,
            "booking_result": booking_result,
            "booking_confirmed": True,
            "next_step": "end"
        }
    except Exception as e:
        return {
            **state,
            "next_step": "end",
            "error": f"Booking error: {str(e)}"
        }


def should_continue(state: WorkflowState) -> Literal["triage", "provider_matching", "scheduling", "human_confirmation", "booking", "end"]:
    """
    Conditional edge function to determine next step.
    """
    next_step = state.get("next_step", "end")
    
    if next_step == "triage":
        return "triage"
    elif next_step == "provider_matching":
        return "provider_matching"
    elif next_step == "scheduling":
        return "scheduling"
    elif next_step == "human_confirmation":
        return "human_confirmation"
    elif next_step == "booking":
        return "booking"
    else:
        return "end"


# Create the workflow graph
def create_workflow():
    """
    Create and return the LangGraph workflow.
    """
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("chat", chat_node)
    workflow.add_node("triage", triage_node)
    workflow.add_node("provider_matching", provider_matching_node)
    workflow.add_node("scheduling", scheduling_node)
    workflow.add_node("human_confirmation", human_confirmation_node)
    workflow.add_node("booking", booking_node)
    
    # Set entry point
    workflow.set_entry_point("chat")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "chat",
        should_continue,
        {
            "triage": "triage",
            "end": END
        }
    )
    
    workflow.add_edge("triage", "provider_matching")
    workflow.add_edge("provider_matching", "scheduling")
    workflow.add_edge("scheduling", "human_confirmation")
    
    workflow.add_conditional_edges(
        "human_confirmation",
        should_continue,
        {
            "booking": "booking",
            "end": END
        }
    )
    
    workflow.add_edge("booking", END)
    
    return workflow.compile()


# Create a workflow that can start from triage
def create_triage_workflow():
    """
    Create a workflow that starts from triage (for medical requests).
    """
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("provider_matching", provider_matching_node)
    workflow.add_node("scheduling", scheduling_node)
    workflow.add_node("human_confirmation", human_confirmation_node)
    workflow.add_node("booking", booking_node)
    
    # Set entry point to triage
    workflow.set_entry_point("triage")
    
    # Add edges
    workflow.add_edge("triage", "provider_matching")
    workflow.add_edge("provider_matching", "scheduling")
    workflow.add_edge("scheduling", "human_confirmation")
    
    workflow.add_conditional_edges(
        "human_confirmation",
        should_continue,
        {
            "booking": "booking",
            "end": END
        }
    )
    
    workflow.add_edge("booking", END)
    
    return workflow.compile()


# Create the workflow instances
app = create_workflow()
triage_app = create_triage_workflow()


def run_workflow(
    user_message: str,
    session_id: str = "default",
    context: dict = None,
    patient_info: dict = None,
    start_from: str = "triage"
) -> dict:
    """
    Run the complete workflow from triage to booking.
    
    Args:
        user_message: User's message/request
        session_id: Session ID for tracking
        context: Additional context (age, gender, etc.)
        patient_info: Patient information for booking (name, contact)
        start_from: Where to start the workflow (defaults to "triage")
                   Note: ChatAgent handles intent detection, so workflow starts from triage
    
    Returns:
        Final workflow state
    """
    # Prepare initial state
    initial_state = {
        "user_message": user_message,
        "session_id": session_id,
        "context": context or {},
        "patient_info": patient_info or {},
        "symptoms": user_message,  # Extract symptoms from message
        "next_step": "triage"
    }
    
    # Always start from triage since ChatAgent already handles intent detection
    # Triage assesses the condition, then workflow continues to provider matching, scheduling, and booking
    final_state = triage_app.invoke(initial_state)
    
    return final_state


if __name__ == "__main__":
    # Example usage
    result = run_workflow(
        user_message="I need to see a cardiologist for chest pain",
        context={"age": 45, "gender": "male"},
        patient_info={"name": "John Doe", "contact": "john@example.com"}
    )
    print(result)

