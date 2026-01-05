from typing import Optional, Dict, Any, List
import os
from datetime import datetime, timedelta
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Import mock knowledge base
import sys
# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from knowledge_base import MOCK_APPOINTMENT_SLOTS
    # Use mock appointment slots from knowledge base
    APPOINTMENT_SLOTS = MOCK_APPOINTMENT_SLOTS
except ImportError:
    # Fallback to hardcoded slots if knowledge base not available
    APPOINTMENT_SLOTS = [
        {
            "slot_id": "slot_001",
            "provider_type": "primary_care",
            "provider_name": "Dr. Smith",
            "date": "2024-01-15",
            "time": "09:00",
            "duration_minutes": 30,
            "available": True,
            "description": "Primary care appointment for routine checkup, general consultation, preventive care, or chronic condition management."
        }
    ]

# Initialize embeddings and vector store (lazy loading)
_vector_store = None
_embeddings = None


def _initialize_vector_store():
    """Initialize the vector store with appointment slot information."""
    global _vector_store, _embeddings
    
    if _vector_store is not None:
        return _vector_store
    
    # Initialize embeddings model
    try:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        print(f"Warning: Could not load embeddings model: {e}")
        _embeddings = None
        return None
    
    # Create documents from appointment slots
    documents = []
    for slot in APPOINTMENT_SLOTS:
        if not slot.get("available", False):
            continue
        
        # Create a comprehensive text description for each slot
        text = f"Appointment with {slot['provider_name']} ({slot['provider_type']}). "
        text += f"{slot['description']} "
        text += f"Date: {slot['date']} at {slot['time']}. "
        text += f"Duration: {slot['duration_minutes']} minutes."
        
        metadata = {
            "slot_id": slot["slot_id"],
            "provider_type": slot["provider_type"],
            "provider_name": slot["provider_name"],
            "date": slot["date"],
            "time": slot["time"],
            "duration_minutes": slot["duration_minutes"]
        }
        
        documents.append(Document(page_content=text, metadata=metadata))
    
    # Create FAISS vector store
    try:
        _vector_store = FAISS.from_documents(documents, _embeddings)
    except Exception as e:
        print(f"Warning: Could not create vector store: {e}")
        return None
    
    return _vector_store


def _extract_scheduling_info(request: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract scheduling preferences from request and context."""
    info = {
        "urgency": "routine",
        "preferred_date": None,
        "preferred_time": None,
        "provider_type": None,
        "duration_needed": None
    }
    
    request_lower = request.lower()
    
    # Extract urgency
    if any(word in request_lower for word in ["urgent", "asap", "soon", "immediate", "emergency"]):
        info["urgency"] = "urgent"
    elif any(word in request_lower for word in ["routine", "regular", "checkup", "follow-up"]):
        info["urgency"] = "routine"
    
    # Extract date preferences
    if "today" in request_lower:
        info["preferred_date"] = datetime.now().strftime("%Y-%m-%d")
    elif "tomorrow" in request_lower:
        info["preferred_date"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "next week" in request_lower:
        info["preferred_date"] = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Extract time preferences
    if "morning" in request_lower or "am" in request_lower:
        info["preferred_time"] = "morning"
    elif "afternoon" in request_lower or "pm" in request_lower:
        info["preferred_time"] = "afternoon"
    elif "evening" in request_lower:
        info["preferred_time"] = "evening"
    
    # Extract from context
    if context:
        if "urgency" in context:
            info["urgency"] = str(context["urgency"]).lower()
        if "preferred_date" in context:
            info["preferred_date"] = context["preferred_date"]
        if "preferred_time" in context:
            info["preferred_time"] = context["preferred_time"]
        if "provider_type" in context:
            info["provider_type"] = context["provider_type"]
        if "duration_needed" in context:
            info["duration_needed"] = context["duration_needed"]
    
    return info


def _filter_slots_by_date(slots: List[Dict[str, Any]], preferred_date: Optional[str]) -> List[Dict[str, Any]]:
    """Filter slots by preferred date."""
    if preferred_date is None:
        return slots
    
    filtered = []
    for slot in slots:
        slot_date = slot.get("metadata", {}).get("date")
        if slot_date == preferred_date:
            filtered.append(slot)
        # Also include slots within 3 days if exact match not found
        elif slot_date:
            try:
                slot_dt = datetime.strptime(slot_date, "%Y-%m-%d")
                pref_dt = datetime.strptime(preferred_date, "%Y-%m-%d")
                if abs((slot_dt - pref_dt).days) <= 3:
                    filtered.append(slot)
            except ValueError:
                pass
    
    return filtered if filtered else slots


def _filter_slots_by_time(slots: List[Dict[str, Any]], preferred_time: Optional[str]) -> List[Dict[str, Any]]:
    """Filter slots by preferred time of day."""
    if preferred_time is None:
        return slots
    
    filtered = []
    for slot in slots:
        slot_time = slot.get("metadata", {}).get("time", "")
        hour = int(slot_time.split(":")[0]) if ":" in slot_time else 12
        
        if preferred_time == "morning" and 6 <= hour < 12:
            filtered.append(slot)
        elif preferred_time == "afternoon" and 12 <= hour < 17:
            filtered.append(slot)
        elif preferred_time == "evening" and 17 <= hour < 21:
            filtered.append(slot)
    
    return filtered if filtered else slots


def _filter_slots_by_provider(slots: List[Dict[str, Any]], provider_type: Optional[str]) -> List[Dict[str, Any]]:
    """Filter slots by provider type."""
    if provider_type is None:
        return slots
    
    filtered = []
    for slot in slots:
        slot_provider = slot.get("metadata", {}).get("provider_type", "")
        if slot_provider == provider_type:
            filtered.append(slot)
    
    return filtered if filtered else slots


def _filter_slots_by_urgency(slots: List[Dict[str, Any]], urgency: str) -> List[Dict[str, Any]]:
    """Filter slots based on urgency level."""
    if urgency == "urgent":
        # For urgent, prioritize same-day or next-day slots
        filtered = []
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        for slot in slots:
            slot_date = slot.get("metadata", {}).get("date", "")
            if slot_date in [today, tomorrow]:
                filtered.append(slot)
        
        return filtered if filtered else slots
    
    return slots


def schedule_appointment(
    request: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Core scheduling logic that finds appropriate appointment slots using RAG-based semantic search.
    
    Args:
        request: Scheduling request description (e.g., "I need to see a cardiologist for chest pain")
        context: Optional context dictionary with additional information 
                 (e.g., urgency, preferred_date, preferred_time, provider_type, duration_needed)
    
    Returns:
        Dictionary containing:
            - available_slots: List of recommended appointment slots
            - recommended_slot: Primary recommended slot
            - reasoning: Explanation for the recommendation
            - request: Original request
            - context: Provided context
    """
    # Extract scheduling preferences
    scheduling_info = _extract_scheduling_info(request, context)
    
    # Initialize vector store
    vector_store = _initialize_vector_store()
    
    # If vector store initialization failed, fall back to basic matching
    if vector_store is None:
        return _fallback_schedule(request, scheduling_info)
    
    # Perform semantic search using RAG
    try:
        # Build search query from request
        search_query = request
        if scheduling_info.get("provider_type"):
            search_query = f"{scheduling_info['provider_type']} appointment: {request}"
        
        # Retrieve top matches
        results = vector_store.similarity_search_with_score(
            search_query,
            k=10  # Get top 10 matches
        )
        
        # Process results
        potential_slots = []
        for doc, score in results:
            slot_data = {
                "slot_id": doc.metadata.get("slot_id"),
                "provider_type": doc.metadata.get("provider_type"),
                "provider_name": doc.metadata.get("provider_name"),
                "date": doc.metadata.get("date"),
                "time": doc.metadata.get("time"),
                "duration_minutes": doc.metadata.get("duration_minutes"),
                "match_score": float(score),  # Lower score = better match in FAISS
                "description": doc.page_content
            }
            potential_slots.append(slot_data)
        
        # Apply filters
        filtered_slots = potential_slots
        
        # Filter by provider type if specified
        if scheduling_info.get("provider_type"):
            provider_type = scheduling_info["provider_type"]
            # Map emergency_room to urgent_care since slots don't have emergency_room type
            if provider_type == "emergency_room":
                provider_type = "urgent_care"
            filtered_slots = _filter_slots_by_provider(filtered_slots, provider_type)
        
        # Filter by urgency
        filtered_slots = _filter_slots_by_urgency(filtered_slots, scheduling_info["urgency"])
        
        # Filter by preferred date
        if scheduling_info.get("preferred_date"):
            filtered_slots = _filter_slots_by_date(filtered_slots, scheduling_info["preferred_date"])
        
        # Filter by preferred time
        if scheduling_info.get("preferred_time"):
            filtered_slots = _filter_slots_by_time(filtered_slots, scheduling_info["preferred_time"])
        
        # Sort by match score (lower is better for FAISS distance)
        filtered_slots.sort(key=lambda x: x.get("match_score", float('inf')))
        
        # Determine recommended slot
        recommended_slot = None
        reasoning = ""
        
        if filtered_slots:
            recommended_slot = filtered_slots[0]
            reasoning = f"Found appointment slot with {recommended_slot['provider_name']} on {recommended_slot['date']} at {recommended_slot['time']}."
            reasoning += f" Matched based on: '{request}'."
            
            if scheduling_info.get("urgency") == "urgent":
                reasoning += " Prioritized for urgent scheduling."
        else:
            # Fallback if filtering removed all results
            if potential_slots:
                recommended_slot = potential_slots[0]
                reasoning = f"Found appointment slot with {recommended_slot['provider_name']} on {recommended_slot['date']} at {recommended_slot['time']}."
            else:
                return _fallback_schedule(request, scheduling_info)
        
        # Format available slots list (top 5)
        available_slots_list = [
            {
                "slot_id": s["slot_id"],
                "provider_type": s["provider_type"],
                "provider_name": s["provider_name"],
                "date": s["date"],
                "time": s["time"],
                "duration_minutes": s["duration_minutes"],
                "match_score": round(1.0 / (1.0 + s.get("match_score", 1.0)), 3)  # Convert distance to similarity score
            }
            for s in filtered_slots[:5]
        ]
        
        # Ensure recommended_slot has all required fields
        if recommended_slot and recommended_slot.get("slot_id"):
            return {
                "available_slots": available_slots_list,
                "recommended_slot": {
                    "slot_id": recommended_slot["slot_id"],
                    "provider_type": recommended_slot.get("provider_type"),
                    "provider_name": recommended_slot.get("provider_name"),
                    "date": recommended_slot.get("date"),
                    "time": recommended_slot.get("time"),
                    "duration_minutes": recommended_slot.get("duration_minutes")
                },
                "reasoning": reasoning,
                "request": request,
                "context": context
            }
        else:
            # If no valid slot found, use fallback
            return _fallback_schedule(request, scheduling_info)
    
    except Exception as e:
        # Fallback if RAG search fails
        print(f"Warning: RAG search failed: {e}")
        return _fallback_schedule(request, scheduling_info)


def _fallback_schedule(
    request: str,
    scheduling_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Fallback scheduling logic when RAG is unavailable."""
    # Find first available slot matching criteria
    available_slots = [s for s in APPOINTMENT_SLOTS if s.get("available", False)]
    
    # Apply basic filters
    if scheduling_info.get("provider_type"):
        provider_type = scheduling_info["provider_type"]
        # Map emergency_room to urgent_care since slots don't have emergency_room type
        if provider_type == "emergency_room":
            provider_type = "urgent_care"
        available_slots = [s for s in available_slots if s["provider_type"] == provider_type]
    
    if scheduling_info.get("urgency") == "urgent":
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        available_slots = [s for s in available_slots if s["date"] in [today, tomorrow]]
    
    if not available_slots:
        available_slots = [s for s in APPOINTMENT_SLOTS if s.get("available", False)]
    
    recommended_slot = available_slots[0] if available_slots else None
    
    if recommended_slot and recommended_slot.get("slot_id"):
        reasoning = f"Found appointment slot with {recommended_slot['provider_name']} on {recommended_slot['date']} at {recommended_slot['time']}. (Fallback mode - RAG unavailable)"
    else:
        # If still no slot, try to find ANY available slot regardless of provider type
        all_available = [s for s in APPOINTMENT_SLOTS if s.get("available", False)]
        if all_available:
            recommended_slot = all_available[0]
            reasoning = f"Found appointment slot with {recommended_slot['provider_name']} on {recommended_slot['date']} at {recommended_slot['time']}. (Fallback mode - using any available slot)"
        else:
            reasoning = "No available slots found. Please try again later."
            recommended_slot = {
                "slot_id": None,
                "provider_type": None,
                "provider_name": None,
                "date": None,
                "time": None,
                "duration_minutes": None
            }
    
    return {
        "available_slots": [{
            "slot_id": s["slot_id"],
            "provider_type": s["provider_type"],
            "provider_name": s["provider_name"],
            "date": s["date"],
            "time": s["time"],
            "duration_minutes": s["duration_minutes"],
            "match_score": 1.0
        } for s in available_slots[:5]],
        "recommended_slot": {
            "slot_id": recommended_slot.get("slot_id"),
            "provider_type": recommended_slot.get("provider_type"),
            "provider_name": recommended_slot.get("provider_name"),
            "date": recommended_slot.get("date"),
            "time": recommended_slot.get("time"),
            "duration_minutes": recommended_slot.get("duration_minutes")
        },
        "reasoning": reasoning,
        "request": request,
        "context": scheduling_info
    }

