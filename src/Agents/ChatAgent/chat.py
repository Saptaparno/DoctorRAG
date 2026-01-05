import re
import sys
import os

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from history import get_history, trim_history
from bot import build_prompt, pipe

# Add src directory to path to import workflow
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from workflow import run_workflow

# Store pending bookings per session (slot info waiting for user confirmation)
_pending_bookings = {}

# Track sessions waiting for patient info (user confirmed but didn't provide details)
_waiting_for_patient_info = set()


def _detect_scheduling_intent(user_text: str) -> bool:
    """Detect if user wants to schedule/book an appointment."""
    scheduling_keywords = [
        "schedule", "book", "appointment", "see a doctor", "make an appointment",
        "need to see", "want to see", "book me", "schedule me", "make appointment",
        "find appointment", "available", "when can i", "i need an appointment"
    ]
    user_lower = user_text.lower()
    return any(keyword in user_lower for keyword in scheduling_keywords)


def _detect_medical_request_intent(user_text: str) -> bool:
    """Detect if user has a medical request/symptoms that needs triage."""
    medical_keywords = [
        "pain", "hurt", "ache", "symptom", "feeling", "unwell", "sick", "ill",
        "fever", "cough", "headache", "nausea", "dizzy", "chest pain", "stomach",
        "rash", "bleeding", "injury", "wound", "infection", "problem", "issue",
        "concern", "worried", "not feeling well", "what's wrong", "diagnosis",
        "condition", "disease", "disorder", "medical", "health", "doctor"
    ]
    user_lower = user_text.lower()
    return any(keyword in user_lower for keyword in medical_keywords)


def _detect_booking_confirmation(user_text: str) -> tuple[bool, dict]:
    """
    Detect if user is confirming a booking with their information.
    Returns (is_confirmation, extracted_patient_info)
    """
    confirmation_keywords = [
        "yes", "confirm", "book it", "proceed", "go ahead", "book this",
        "i confirm", "please book", "book me", "yes please", "sure", "okay"
    ]
    user_lower = user_text.lower()
    is_confirmation = any(keyword in user_lower for keyword in confirmation_keywords)
    
    # Extract patient info from message (name, email, phone)
    patient_info = {}
    
    # Try to extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, user_text)
    if emails:
        patient_info["contact"] = emails[0]
    
    # Try to extract phone (various formats)
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
        r'\b\d{10}\b',  # 10 digits
        r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'  # International
    ]
    for pattern in phone_patterns:
        phones = re.findall(pattern, user_text)
        if phones and "contact" not in patient_info:
            patient_info["contact"] = phones[0]
            break
    
    # Try to extract name (look for "I'm", "my name is", "this is", etc.)
    name_patterns = [
        r"(?:i'?m|i am|my name is|this is|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    ]
    for pattern in name_patterns:
        matches = re.findall(pattern, user_text, re.IGNORECASE)
        if matches:
            patient_info["name"] = matches[0].strip()
            break
    
    return is_confirmation, patient_info


def _format_workflow_response(workflow_result: dict, session_id: str = "default") -> str:
    """Format workflow result into a user-friendly chat response."""
    error = workflow_result.get("error")
    if error:
        return f"I encountered an issue: {error}. Please try again or provide more details."
    
    # Check if we have scheduling results
    scheduling_result = workflow_result.get("scheduling_result")
    if scheduling_result:
        recommended_slot = scheduling_result.get("recommended_slot", {})
        if recommended_slot:
            provider_name = recommended_slot.get("provider_name", "provider")
            date = recommended_slot.get("date", "")
            time = recommended_slot.get("time", "")
            
            # Store pending booking in session for confirmation
            _pending_bookings[session_id] = recommended_slot
            
            response = f"I found an available appointment for you:\n\n"
            response += f"**Provider:** {provider_name}\n"
            response += f"**Date:** {date}\n"
            response += f"**Time:** {time}\n\n"
            response += "Would you like me to book this appointment? Please reply with 'yes' or 'confirm' along with your name and contact information (email or phone)."
            return response
    
    # Check if booking was successful
    booking_result = workflow_result.get("booking_result")
    if booking_result:
        booking_id = booking_result.get("booking_id", "")
        confirmation_code = booking_result.get("confirmation_code", "")
        appointment = booking_result.get("appointment", {})
        
        response = f"✅ Appointment booked successfully!\n\n"
        response += f"**Booking ID:** {booking_id}\n"
        response += f"**Confirmation Code:** {confirmation_code}\n"
        response += f"**Provider:** {appointment.get('provider_name', '')}\n"
        response += f"**Date:** {appointment.get('date', '')}\n"
        response += f"**Time:** {appointment.get('time', '')}\n\n"
        response += "Please save your confirmation code for your records."
        return response
    
    # Check triage result
    triage_result = workflow_result.get("triage_result")
    if triage_result:
        priority = triage_result.get("priority", "")
        assessment = triage_result.get("assessment", "")
        if priority and assessment:
            response = f"Based on your symptoms, I've assessed your condition:\n\n"
            response += f"**Priority:** {priority.upper()}\n"
            response += f"**Assessment:** {assessment}\n\n"
            response += "Let me find an appropriate provider and available appointment slots for you..."
            return response
    
    return "I'm processing your request. Please wait..."


def chatflow(user_text, session_id="default", context=None, patient_info=None):
    """
    Enhanced chatflow that detects scheduling intent or medical requests and triggers workflow.
    Also handles booking confirmation in the chat flow.
    
    Args:
        user_text: User's message
        session_id: Session ID
        context: Optional context (age, gender, etc.)
        patient_info: Optional patient info (name, contact) for booking
    
    Returns:
        Chat response (either from chatbot or workflow)
    """
    # Handle pending booking confirmation or patient info collection
    if session_id in _pending_bookings:
        pending_slot = _pending_bookings[session_id]
        
        # Check if user is confirming (with keywords) or just providing info
        is_confirmation, extracted_info = _detect_booking_confirmation(user_text)
        
        # If we're waiting for patient info, user might just be providing it without confirmation keywords
        # So we should extract info more aggressively
        if session_id in _waiting_for_patient_info:
            # User is providing info after we asked for it
            # Extract info even if no confirmation keywords were detected
            # The extraction function will still try to find email, phone, name
            _, extracted_info = _detect_booking_confirmation(user_text)
            
            # Also try to extract info from any text (not just confirmation messages)
            # This handles cases like "my email is john@example.com" without "yes"
            # Extract email if not already found
            if "contact" not in extracted_info:
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                emails = re.findall(email_pattern, user_text)
                if emails:
                    extracted_info["contact"] = emails[0]
            
            # Extract phone if not already found
            if "contact" not in extracted_info:
                phone_patterns = [
                    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                    r'\b\d{10}\b',
                    r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
                ]
                for pattern in phone_patterns:
                    phones = re.findall(pattern, user_text)
                    if phones:
                        extracted_info["contact"] = phones[0]
                        break
            
            # Extract name if not already found
            if "name" not in extracted_info:
                name_patterns = [
                    r"(?:i'?m|i am|my name is|this is|call me|name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                    r"name[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
                ]
                for pattern in name_patterns:
                    matches = re.findall(pattern, user_text, re.IGNORECASE)
                    if matches:
                        extracted_info["name"] = matches[0].strip()
                        break
        
        # Merge extracted patient info with provided patient_info
        final_patient_info = {**(patient_info or {}), **extracted_info}
        
        # Check if we have required contact info
        if not final_patient_info.get("contact"):
            # Missing contact info - ask for it
            if is_confirmation or session_id in _waiting_for_patient_info:
                # User confirmed or is responding to our request, but still no contact info
                _waiting_for_patient_info.add(session_id)
                return "I'd be happy to book that appointment! However, I need your contact information to complete the booking. Please provide your name and either your email address or phone number."
            else:
                # User might be providing info but we didn't extract it - check if they're just chatting
                # If there's a pending booking, they're probably trying to provide info
                _waiting_for_patient_info.add(session_id)
                return "To complete your booking, I need your contact information (email or phone number). Please provide it."
        
        # We have contact info - proceed with booking
        # Clear waiting status
        _waiting_for_patient_info.discard(session_id)
        
        # Call booking confirmation endpoint
        try:
            import requests
            chatagent_url = os.getenv("CHATAGENT_API_URL", "http://localhost:8001")
            booking_url = f"{chatagent_url}/booking/confirm"
            
            booking_payload = {
                "slot_id": pending_slot.get("slot_id"),
                "patient_name": final_patient_info.get("name", "Unknown"),
                "patient_contact": final_patient_info.get("contact", ""),
                "appointment_details": {
                    "provider_type": pending_slot.get("provider_type"),
                    "provider_name": pending_slot.get("provider_name"),
                    "date": pending_slot.get("date"),
                    "time": pending_slot.get("time"),
                    "duration_minutes": pending_slot.get("duration_minutes")
                },
                "additional_info": final_patient_info
            }
            
            response = requests.post(booking_url, json=booking_payload, timeout=30)
            response.raise_for_status()
            booking_result = response.json()
            
            # Clear pending booking
            del _pending_bookings[session_id]
            
            # Format success response
            confirmation_code = booking_result.get("confirmation_code", "")
            booking_id = booking_result.get("booking_id", "")
            appointment = booking_result.get("appointment", {})
            
            reply = f"✅ Appointment booked successfully!\n\n"
            reply += f"**Booking ID:** {booking_id}\n"
            reply += f"**Confirmation Code:** {confirmation_code}\n"
            reply += f"**Provider:** {appointment.get('provider_name', '')}\n"
            reply += f"**Date:** {appointment.get('date', '')}\n"
            reply += f"**Time:** {appointment.get('time', '')}\n\n"
            reply += "Please save your confirmation code for your records."
            
            # Add to history
            history = get_history(session_id)
            history.add_user_message(user_text)
            history.add_ai_message(reply)
            
            return reply
        except Exception as e:
            print(f"Booking confirmation error: {e}")
            return f"I encountered an error while booking your appointment: {str(e)}. Please try again."
    
    # Detect medical request or scheduling intent (both start from triage)
    if _detect_medical_request_intent(user_text) or _detect_scheduling_intent(user_text):
        try:
            # Run workflow starting from triage
            # ChatAgent already detected intent, so we skip the chat node
            # and go straight to triage to assess the condition/priority
            workflow_result = run_workflow(
                user_message=user_text,
                session_id=session_id,
                context=context or {},
                patient_info=patient_info or {},
                start_from="triage"
            )
            
            # Format workflow response
            reply = _format_workflow_response(workflow_result, session_id=session_id)
            
            # Add to history
            history = get_history(session_id)
            history.add_user_message(user_text)
            history.add_ai_message(reply)
            
            return reply
        except Exception as e:
            # Fallback to normal chat if workflow fails
            print(f"Workflow error: {e}")
            # Continue to normal chat flow below
    
    # Normal chat flow
    history = get_history(session_id)
    trim_history(session_id)
    prompt = build_prompt(history, user_text)
    out = pipe(prompt)[0]["generated_text"]
    
    # Extract reply (should not contain prompt since return_full_text=False)
    if out.startswith(prompt):
        reply = out[len(prompt):].strip()
    else:
        reply = out.strip()
    
    # Clean up the reply
    # Remove reasoning blocks
    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL)
    
    # Stop at special tokens
    if "<|im_end|>" in reply:
        reply = reply.split("<|im_end|>")[0].strip()
    
    # Stop if model starts generating user or assistant tags (continuing conversation incorrectly)
    if "\nUser:" in reply:
        reply = reply.split("\nUser:")[0].strip()
    if "\nAssistant:" in reply:
        reply = reply.split("\nAssistant:")[0].strip()
    
    # If reply seems to be rambling (very long, multiple paragraphs), take first reasonable part
    # This handles cases where model generates unrelated continuation text
    if len(reply) > 500:  # If reply is very long
        # Try to find a natural stopping point
        sentences = reply.split('. ')
        if len(sentences) > 5:  # If there are many sentences
            # Take first few sentences that make sense
            reasonable_reply = '. '.join(sentences[:3]) + '.'
            if len(reasonable_reply) > 50:  # Only use if it's substantial
                reply = reasonable_reply
    
    # Final cleanup
    reply = reply.strip()
    
    # Debug logging
    print(f"[chat.py] User input: {user_text}")
    print(f"[chat.py] Extracted reply (length: {len(reply)}): {reply[:200]}...")
    
    history.add_user_message(user_text)
    history.add_ai_message(reply)
    return reply