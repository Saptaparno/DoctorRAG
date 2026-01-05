from typing import Optional, Dict, Any
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ChatAgent API configuration
CHATAGENT_API_URL = os.getenv("CHATAGENT_API_URL", "http://localhost:8001")


def book_appointment(
    slot_id: str,
    patient_name: str,
    patient_contact: str,
    appointment_details: Dict[str, Any],
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main booking function that creates an appointment booking.
    Uses ChatAgent's /booking/confirm endpoint.
    
    Args:
        slot_id: The appointment slot ID to book
        patient_name: Patient's name
        patient_contact: Patient's contact information (phone/email)
        appointment_details: Appointment details from scheduling
        additional_info: Optional additional patient information
    
    Returns:
        Dictionary containing booking confirmation details
    """
    try:
        # Call ChatAgent's booking confirmation endpoint
        url = f"{CHATAGENT_API_URL}/booking/confirm"
        
        payload = {
            "slot_id": slot_id,
            "patient_name": patient_name,
            "patient_contact": patient_contact,
            "appointment_details": appointment_details,
            "additional_info": additional_info
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Booking API request failed: {str(e)}")

