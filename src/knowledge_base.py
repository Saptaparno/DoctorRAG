"""
Mock Knowledge Base for testing the DoctorRAG application.
Contains sample data for providers, appointment slots, and test scenarios.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any


# Mock Provider Database
MOCK_PROVIDERS = [
    {
        "provider_id": "prov_001",
        "name": "Dr. Sarah Smith",
        "type": "primary_care",
        "specialties": ["general medicine", "preventive care", "chronic conditions"],
        "location": "123 Main St, Medical Center",
        "phone": "(555) 123-4567",
        "email": "sarah.smith@hospital.com",
        "availability": "Monday-Friday, 9AM-5PM",
        "accepting_new_patients": True
    },
    {
        "provider_id": "prov_002",
        "name": "Dr. Michael Johnson",
        "type": "cardiologist",
        "specialties": ["cardiology", "heart disease", "hypertension", "arrhythmia"],
        "location": "456 Heart Ave, Cardiology Wing",
        "phone": "(555) 234-5678",
        "email": "michael.johnson@hospital.com",
        "availability": "Monday-Friday, 8AM-4PM",
        "accepting_new_patients": True
    },
    {
        "provider_id": "prov_003",
        "name": "Dr. Emily Williams",
        "type": "dermatologist",
        "specialties": ["dermatology", "skin conditions", "acne", "skin cancer"],
        "location": "789 Skin Blvd, Dermatology Clinic",
        "phone": "(555) 345-6789",
        "email": "emily.williams@hospital.com",
        "availability": "Monday-Friday, 10AM-6PM",
        "accepting_new_patients": True
    },
    {
        "provider_id": "prov_004",
        "name": "Dr. James Brown",
        "type": "pediatrician",
        "specialties": ["pediatrics", "child health", "vaccinations", "growth monitoring"],
        "location": "321 Kids Lane, Pediatric Center",
        "phone": "(555) 456-7890",
        "email": "james.brown@hospital.com",
        "availability": "Monday-Friday, 8AM-5PM",
        "accepting_new_patients": True,
        "age_range": (0, 18)
    },
    {
        "provider_id": "prov_005",
        "name": "Dr. Robert Davis",
        "type": "orthopedist",
        "specialties": ["orthopedics", "fractures", "joint pain", "sports injuries"],
        "location": "654 Bone St, Orthopedic Center",
        "phone": "(555) 567-8901",
        "email": "robert.davis@hospital.com",
        "availability": "Monday-Friday, 9AM-5PM",
        "accepting_new_patients": True
    },
    {
        "provider_id": "prov_006",
        "name": "Dr. Lisa Miller",
        "type": "psychiatrist",
        "specialties": ["psychiatry", "mental health", "depression", "anxiety"],
        "location": "987 Mind Ave, Mental Health Center",
        "phone": "(555) 678-9012",
        "email": "lisa.miller@hospital.com",
        "availability": "Monday-Friday, 9AM-5PM",
        "accepting_new_patients": True
    },
    {
        "provider_id": "prov_007",
        "name": "Dr. Jennifer Wilson",
        "type": "gynecologist",
        "specialties": ["gynecology", "women's health", "pregnancy", "reproductive health"],
        "location": "147 Women's Health Dr, Gynecology Clinic",
        "phone": "(555) 789-0123",
        "email": "jennifer.wilson@hospital.com",
        "availability": "Monday-Friday, 8AM-4PM",
        "accepting_new_patients": True
    },
    {
        "provider_id": "urgent_001",
        "name": "Urgent Care Center",
        "type": "urgent_care",
        "specialties": ["urgent care", "minor injuries", "infections", "fevers"],
        "location": "555 Urgent Way, Urgent Care Building",
        "phone": "(555) 890-1234",
        "email": "urgentcare@hospital.com",
        "availability": "Monday-Sunday, 8AM-8PM",
        "accepting_new_patients": True
    },
    {
        "provider_id": "er_001",
        "name": "Emergency Room",
        "type": "emergency_room",
        "specialties": ["emergency", "trauma", "critical care", "life-threatening"],
        "location": "999 Emergency Blvd, Main Hospital",
        "phone": "(555) 911-0000",
        "email": "emergency@hospital.com",
        "availability": "24/7",
        "accepting_new_patients": True
    }
]


def generate_mock_appointment_slots(days_ahead: int = 30) -> List[Dict[str, Any]]:
    """
    Generate mock appointment slots for the next N days.
    
    Args:
        days_ahead: Number of days ahead to generate slots
    
    Returns:
        List of appointment slot dictionaries
    """
    slots = []
    base_date = datetime.now()
    slot_id_counter = 1
    
    # Generate slots for each provider
    for provider in MOCK_PROVIDERS:
        provider_type = provider["type"]
        provider_name = provider["name"]
        
        # Skip emergency room (walk-in only)
        if provider_type == "emergency_room":
            continue
        
        # Generate slots for the next N days
        for day_offset in range(days_ahead):
            slot_date = base_date + timedelta(days=day_offset)
            date_str = slot_date.strftime("%Y-%m-%d")
            
            # Skip weekends for most providers (except urgent care)
            if slot_date.weekday() >= 5 and provider_type != "urgent_care":
                continue
            
            # Generate time slots based on provider type
            if provider_type == "urgent_care":
                # Urgent care: every 30 minutes from 8AM to 8PM
                time_slots = [
                    f"{hour:02d}:{minute:02d}"
                    for hour in range(8, 20)
                    for minute in [0, 30]
                ]
                duration = 20
            else:
                # Regular providers: hourly slots from 9AM to 5PM
                time_slots = [f"{hour:02d}:00" for hour in range(9, 17)]
                duration = 30 if provider_type == "primary_care" else 45
            
            # Create slots for each time
            for time_slot in time_slots:
                slot = {
                    "slot_id": f"slot_{slot_id_counter:03d}",
                    "provider_id": provider["provider_id"],
                    "provider_type": provider_type,
                    "provider_name": provider_name,
                    "date": date_str,
                    "time": time_slot,
                    "duration_minutes": duration,
                    "available": True,
                    "description": f"{provider_name} appointment for {provider_type} care. {provider.get('specialties', [])[0] if provider.get('specialties') else 'general consultation'}."
                }
                slots.append(slot)
                slot_id_counter += 1
    
    return slots


# Pre-generated slots for testing
MOCK_APPOINTMENT_SLOTS = generate_mock_appointment_slots(30)


# Mock Patient Test Scenarios
MOCK_TEST_SCENARIOS = [
    {
        "scenario_id": "test_001",
        "name": "Emergency - Chest Pain",
        "user_message": "I'm having severe chest pain and difficulty breathing",
        "context": {"age": 45, "gender": "male", "temperature": 98.6},
        "expected_priority": "emergency",
        "expected_provider_type": "emergency_room"
    },
    {
        "scenario_id": "test_002",
        "name": "Urgent - High Fever",
        "user_message": "I have a high fever of 103 degrees and severe headache",
        "context": {"age": 32, "gender": "female", "temperature": 103.0},
        "expected_priority": "urgent",
        "expected_provider_type": "urgent_care"
    },
    {
        "scenario_id": "test_003",
        "name": "Routine - Skin Rash",
        "user_message": "I have a mild skin rash on my arm that's been there for a few days",
        "context": {"age": 28, "gender": "non-binary"},
        "expected_priority": "routine",
        "expected_provider_type": "dermatologist"
    },
    {
        "scenario_id": "test_004",
        "name": "Pediatric - Child Fever",
        "user_message": "My 5-year-old child has a fever and is not eating",
        "context": {"age": 5, "gender": "male", "temperature": 101.5},
        "expected_priority": "urgent",
        "expected_provider_type": "pediatrician"
    },
    {
        "scenario_id": "test_005",
        "name": "Scheduling - Book Appointment",
        "user_message": "I need to book an appointment for a routine checkup",
        "context": {"age": 40, "gender": "female"},
        "expected_priority": "routine",
        "expected_provider_type": "primary_care"
    },
    {
        "scenario_id": "test_006",
        "name": "Cardiac - Heart Concerns",
        "user_message": "I've been experiencing chest pain and irregular heartbeat",
        "context": {"age": 55, "gender": "male", "existing_conditions": "hypertension"},
        "expected_priority": "urgent",
        "expected_provider_type": "cardiologist"
    },
    {
        "scenario_id": "test_007",
        "name": "Mental Health - Depression",
        "user_message": "I've been feeling very depressed and anxious lately",
        "context": {"age": 35, "gender": "female"},
        "expected_priority": "routine",
        "expected_provider_type": "psychiatrist"
    },
    {
        "scenario_id": "test_008",
        "name": "Orthopedic - Broken Bone",
        "user_message": "I think I broke my arm, it's very painful and I can't move it",
        "context": {"age": 25, "gender": "male", "pain_level": 8},
        "expected_priority": "urgent",
        "expected_provider_type": "orthopedist"
    }
]


# Mock Booking History
MOCK_BOOKINGS = []


def get_mock_provider(provider_id: str) -> Dict[str, Any]:
    """Get a mock provider by ID."""
    for provider in MOCK_PROVIDERS:
        if provider["provider_id"] == provider_id:
            return provider
    return None


def get_mock_providers_by_type(provider_type: str) -> List[Dict[str, Any]]:
    """Get all mock providers of a specific type."""
    return [p for p in MOCK_PROVIDERS if p["type"] == provider_type]


def get_mock_slots_by_provider(provider_id: str) -> List[Dict[str, Any]]:
    """Get all available slots for a provider."""
    return [s for s in MOCK_APPOINTMENT_SLOTS if s["provider_id"] == provider_id and s["available"]]


def get_mock_slots_by_type(provider_type: str) -> List[Dict[str, Any]]:
    """Get all available slots for a provider type."""
    return [s for s in MOCK_APPOINTMENT_SLOTS if s["provider_type"] == provider_type and s["available"]]


def get_mock_slot(slot_id: str) -> Dict[str, Any]:
    """Get a mock appointment slot by ID."""
    for slot in MOCK_APPOINTMENT_SLOTS:
        if slot["slot_id"] == slot_id:
            return slot
    return None


def mark_slot_unavailable(slot_id: str):
    """Mark an appointment slot as unavailable (for booking)."""
    for slot in MOCK_APPOINTMENT_SLOTS:
        if slot["slot_id"] == slot_id:
            slot["available"] = False
            break


def get_test_scenario(scenario_id: str) -> Dict[str, Any]:
    """Get a test scenario by ID."""
    for scenario in MOCK_TEST_SCENARIOS:
        if scenario["scenario_id"] == scenario_id:
            return scenario
    return None


def get_all_test_scenarios() -> List[Dict[str, Any]]:
    """Get all test scenarios."""
    return MOCK_TEST_SCENARIOS


# Export for use in other modules
__all__ = [
    "MOCK_PROVIDERS",
    "MOCK_APPOINTMENT_SLOTS",
    "MOCK_TEST_SCENARIOS",
    "MOCK_BOOKINGS",
    "generate_mock_appointment_slots",
    "get_mock_provider",
    "get_mock_providers_by_type",
    "get_mock_slots_by_provider",
    "get_mock_slots_by_type",
    "get_mock_slot",
    "mark_slot_unavailable",
    "get_test_scenario",
    "get_all_test_scenarios"
]



