from typing import Optional, Dict, Any, List
import re


# Emergency keywords - requires immediate medical attention
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "cardiac arrest", "stopped breathing",
    "difficulty breathing", "can't breathe", "choking", "severe bleeding",
    "unconscious", "unresponsive", "severe allergic reaction", "anaphylaxis",
    "stroke", "seizure", "severe head injury", "severe trauma",
    "severe burn", "overdose", "poisoning", "suicidal", "self-harm"
]

# Urgent keywords - requires prompt medical attention (within hours)
URGENT_KEYWORDS = [
    "high fever", "severe pain", "severe headache", "severe abdominal pain",
    "broken bone", "fracture", "severe vomiting", "severe diarrhea",
    "severe dehydration", "severe infection", "worsening condition",
    "cannot urinate", "severe allergic reaction", "moderate bleeding"
]

# Routine keywords - can wait for regular appointment
ROUTINE_KEYWORDS = [
    "mild", "minor", "checkup", "routine", "follow-up", "prescription refill",
    "cold symptoms", "mild cough", "mild headache", "mild pain"
]


def _check_keywords(text: str, keywords: List[str]) -> bool:
    """Check if any keyword appears in the text (case-insensitive)."""
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def _extract_vital_signs(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract and validate vital signs from context."""
    vitals = {}
    if context:
        if "temperature" in context or "fever" in context:
            temp = context.get("temperature") or context.get("fever", 0)
            try:
                temp = float(temp)
                vitals["temperature"] = temp
                vitals["has_fever"] = temp >= 100.4  # Fahrenheit
            except (ValueError, TypeError):
                pass
        
        if "age" in context:
            try:
                vitals["age"] = int(context["age"])
            except (ValueError, TypeError):
                pass
        
        if "pain_level" in context:
            try:
                pain = int(context["pain_level"])
                vitals["pain_level"] = pain
                vitals["severe_pain"] = pain >= 7  # On scale of 1-10
            except (ValueError, TypeError):
                pass
    
    return vitals


def triage(symptoms: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Core triage logic node that assesses patient symptoms and determines care priority.
    Uses rule-based logic without requiring model inference.
    
    Args:
        symptoms: Patient symptoms or medical inquiry description
        context: Optional context dictionary with additional information 
                 (e.g., age, temperature, pain_level, existing conditions)
    
    Returns:
        Dictionary containing:
            - priority: Priority level ("emergency", "urgent", "routine", or "unknown")
            - assessment: Triage assessment/guidance
            - symptoms: Original symptoms
            - context: Provided context
            - recommended_action: Recommended course of action
    """
    symptoms_lower = symptoms.lower()
    vitals = _extract_vital_signs(context)
    
    # Determine priority based on keywords and vital signs
    priority = "unknown"
    assessment = ""
    recommended_action = ""
    
    # Check for emergency conditions
    if _check_keywords(symptoms, EMERGENCY_KEYWORDS):
        priority = "emergency"
        assessment = "Emergency condition detected. Immediate medical attention required."
        recommended_action = "Call 911 or go to the nearest emergency room immediately."
    
    # Check for urgent conditions
    elif _check_keywords(symptoms, URGENT_KEYWORDS) or vitals.get("severe_pain", False):
        priority = "urgent"
        assessment = "Urgent condition detected. Prompt medical attention recommended within hours."
        recommended_action = "Seek urgent care or visit emergency room if symptoms worsen."
    
    # Check for high fever (urgent if very high)
    elif vitals.get("has_fever", False) and vitals.get("temperature", 0) >= 103:
        priority = "urgent"
        assessment = f"High fever detected ({vitals['temperature']}°F). Medical attention recommended."
        recommended_action = "Seek medical care, especially if fever persists or other symptoms develop."
    
    # Check for routine conditions
    elif _check_keywords(symptoms, ROUTINE_KEYWORDS):
        priority = "routine"
        assessment = "Routine condition. Non-urgent medical attention."
        recommended_action = "Schedule a regular appointment or consult with a healthcare provider."
    
    # Default assessment for unknown cases
    else:
        priority = "unknown"
        assessment = "Unable to determine priority from provided information. Additional assessment may be needed."
        recommended_action = "Consult with a healthcare provider for proper evaluation."
    
    # Add context-specific notes
    if context:
        notes = []
        if "age" in vitals:
            age = vitals["age"]
            if age < 2:
                notes.append("Patient is an infant - may require pediatric-specific care.")
            elif age >= 65:
                notes.append("Patient is elderly - may require additional monitoring.")
        
        if "existing_conditions" in context:
            notes.append(f"Existing conditions: {context['existing_conditions']}")
        
        if notes:
            assessment += " " + " ".join(notes)
    
    return {
        "priority": priority,
        "assessment": assessment,
        "recommended_action": recommended_action,
        "symptoms": symptoms,
        "context": context
    }
