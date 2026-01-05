from typing import Optional, Dict, Any, List
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Provider types and their detailed descriptions for RAG
PROVIDER_TYPES = {
    "emergency_room": {
        "name": "Emergency Room",
        "description": "Emergency room provides immediate medical care for life-threatening conditions, severe injuries, cardiac events, strokes, severe allergic reactions, trauma, and critical emergencies. Available 24/7, no appointment needed.",
        "specialties": ["emergency", "trauma", "critical care", "life-threatening", "cardiac arrest", "stroke", "severe injury"],
        "availability": "24/7",
        "appointment_required": False
    },
    "urgent_care": {
        "name": "Urgent Care",
        "description": "Urgent care centers handle non-life-threatening urgent medical conditions such as fractures, sprains, minor injuries, infections, high fevers, severe pain, and conditions requiring prompt attention but not emergency care. Extended hours, walk-in available.",
        "specialties": ["urgent", "non-life-threatening", "fractures", "minor injuries", "infections", "sprains", "fevers"],
        "availability": "Extended hours",
        "appointment_required": False
    },
    "primary_care": {
        "name": "Primary Care Physician",
        "description": "Primary care physicians provide general medical care, routine checkups, preventive care, management of chronic conditions, health screenings, vaccinations, and general wellness. Regular business hours, appointment required.",
        "specialties": ["general", "routine", "preventive", "chronic conditions", "checkups", "wellness", "general medicine"],
        "availability": "Business hours",
        "appointment_required": True
    },
    "pediatrician": {
        "name": "Pediatrician",
        "description": "Pediatricians specialize in medical care for infants, children, and adolescents from birth to age 18. They handle childhood illnesses, developmental issues, vaccinations, growth monitoring, and pediatric-specific conditions.",
        "specialties": ["pediatric", "children", "infants", "adolescents", "child health", "pediatric medicine"],
        "availability": "Business hours",
        "appointment_required": True,
        "age_range": (0, 18)
    },
    "cardiologist": {
        "name": "Cardiologist",
        "description": "Cardiologists specialize in heart and cardiovascular conditions including chest pain, heart disease, arrhythmias, hypertension, heart attacks, cardiac rehabilitation, and cardiovascular health.",
        "specialties": ["heart", "cardiac", "chest pain", "cardiovascular", "heart disease", "arrhythmia", "hypertension"],
        "availability": "Business hours",
        "appointment_required": True
    },
    "dermatologist": {
        "name": "Dermatologist",
        "description": "Dermatologists specialize in skin conditions including rashes, acne, moles, skin cancer, dermatitis, eczema, psoriasis, hair and nail disorders, and cosmetic dermatology.",
        "specialties": ["skin", "rash", "acne", "dermatology", "moles", "skin cancer", "dermatitis", "eczema"],
        "availability": "Business hours",
        "appointment_required": True
    },
    "orthopedist": {
        "name": "Orthopedist",
        "description": "Orthopedists specialize in bone, joint, and musculoskeletal conditions including fractures, broken bones, sprains, dislocations, joint pain, arthritis, sports injuries, and orthopedic surgery.",
        "specialties": ["bone", "fracture", "joint", "orthopedic", "broken bone", "sprain", "musculoskeletal", "arthritis"],
        "availability": "Business hours",
        "appointment_required": True
    },
    "psychiatrist": {
        "name": "Psychiatrist",
        "description": "Psychiatrists specialize in mental health conditions including depression, anxiety, bipolar disorder, psychiatric disorders, suicidal thoughts, self-harm, mental health crises, and psychiatric medication management.",
        "specialties": ["mental health", "depression", "anxiety", "psychiatric", "suicidal", "mental illness", "bipolar"],
        "availability": "Business hours",
        "appointment_required": True
    },
    "gynecologist": {
        "name": "Gynecologist",
        "description": "Gynecologists specialize in women's health including gynecological conditions, pregnancy care, reproductive health, menstrual issues, pelvic pain, and women's reproductive system health.",
        "specialties": ["women's health", "gynecological", "pregnancy", "reproductive", "menstrual", "pelvic"],
        "availability": "Business hours",
        "appointment_required": True
    }
}

# Initialize embeddings and vector store (lazy loading)
_vector_store = None
_embeddings = None


def _initialize_vector_store():
    """Initialize the vector store with provider information."""
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
        # Fallback to a simpler approach if embeddings fail
        print(f"Warning: Could not load embeddings model: {e}")
        _embeddings = None
        return None
    
    # Create documents from provider information
    documents = []
    for provider_type, provider_info in PROVIDER_TYPES.items():
        # Create a comprehensive text description for each provider
        text = f"{provider_info['name']}. {provider_info['description']} "
        text += f"Specialties: {', '.join(provider_info['specialties'])}. "
        text += f"Availability: {provider_info['availability']}. "
        text += f"Appointment required: {provider_info['appointment_required']}."
        
        metadata = {
            "type": provider_type,
            "name": provider_info["name"],
            "availability": provider_info["availability"],
            "appointment_required": provider_info["appointment_required"]
        }
        
        if "age_range" in provider_info:
            metadata["age_range"] = provider_info["age_range"]
        
        documents.append(Document(page_content=text, metadata=metadata))
    
    # Create FAISS vector store
    try:
        _vector_store = FAISS.from_documents(documents, _embeddings)
    except Exception as e:
        print(f"Warning: Could not create vector store: {e}")
        return None
    
    return _vector_store


def _extract_patient_info(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract and validate patient information from context."""
    info = {}
    if context:
        if "age" in context:
            try:
                info["age"] = int(context["age"])
            except (ValueError, TypeError):
                pass
        
        if "gender" in context:
            info["gender"] = str(context["gender"]).lower()
        
        if "priority" in context:
            info["priority"] = str(context["priority"]).lower()
        
        if "insurance" in context:
            info["insurance"] = context["insurance"]
        
        if "location" in context:
            info["location"] = context["location"]
    
    return info


def _filter_providers_by_age(providers: List[Dict[str, Any]], age: Optional[int]) -> List[Dict[str, Any]]:
    """Filter providers based on patient age."""
    if age is None:
        return providers
    
    filtered = []
    for provider in providers:
        if "age_range" in provider.get("metadata", {}):
            min_age, max_age = provider["metadata"]["age_range"]
            if min_age <= age <= max_age:
                filtered.append(provider)
        else:
            # Providers without age restrictions can see all ages
            filtered.append(provider)
    
    return filtered


def _filter_providers_by_priority(providers: List[Dict[str, Any]], priority: Optional[str]) -> List[Dict[str, Any]]:
    """Filter providers based on priority level."""
    if priority is None:
        return providers
    
    filtered = []
    for provider in providers:
        provider_type = provider.get("metadata", {}).get("type", "")
        
        if priority == "emergency":
            if provider_type == "emergency_room":
                filtered.append(provider)
        elif priority == "urgent":
            if provider_type in ["emergency_room", "urgent_care"]:
                filtered.append(provider)
        else:  # routine or unknown
            if provider_type not in ["emergency_room"]:
                filtered.append(provider)
    
    return filtered


def match_provider(
    symptoms: str,
    priority: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Core provider matching logic that matches patients with appropriate healthcare providers.
    Uses RAG-based semantic search for intelligent provider matching.
    
    Args:
        symptoms: Patient symptoms or medical inquiry description
        priority: Priority level from triage ("emergency", "urgent", "routine", or "unknown")
        context: Optional context dictionary with additional information 
                 (e.g., age, gender, insurance, location)
    
    Returns:
        Dictionary containing:
            - matched_providers: List of recommended provider types
            - primary_provider: Primary recommended provider type
            - reasoning: Explanation for the match
            - symptoms: Original symptoms
            - priority: Priority level
            - context: Provided context
    """
    patient_info = _extract_patient_info(context)
    
    # Override priority from context if provided
    if "priority" in patient_info:
        priority = patient_info["priority"]
    
    # Initialize vector store
    vector_store = _initialize_vector_store()
    
    # If vector store initialization failed, fall back to default recommendation
    if vector_store is None:
        return _fallback_match(symptoms, priority, patient_info)
    
    # Perform semantic search using RAG
    try:
        # Search for relevant providers based on symptoms
        search_query = symptoms
        if priority:
            search_query = f"{priority} priority: {symptoms}"
        
        # Retrieve top matches
        results = vector_store.similarity_search_with_score(
            search_query,
            k=10  # Get top 10 matches
        )
        
        # Process results
        potential_providers = []
        for doc, score in results:
            provider_data = {
                "type": doc.metadata.get("type"),
                "name": doc.metadata.get("name"),
                "availability": doc.metadata.get("availability"),
                "appointment_required": doc.metadata.get("appointment_required"),
                "metadata": doc.metadata,
                "match_score": float(score),  # Lower score = better match in FAISS
                "description": doc.page_content
            }
            
            if "age_range" in doc.metadata:
                provider_data["age_range"] = doc.metadata["age_range"]
            
            potential_providers.append(provider_data)
        
        # Filter by priority
        priority_filtered = _filter_providers_by_priority(potential_providers, priority)
        
        # Filter by age if applicable
        age = patient_info.get("age")
        age_filtered = _filter_providers_by_age(priority_filtered, age)
        
        # Sort by match score (lower is better for FAISS distance)
        age_filtered.sort(key=lambda x: x.get("match_score", float('inf')))
        
        # Determine primary provider
        primary_provider = None
        reasoning = ""
        
        if age_filtered:
            primary_provider = age_filtered[0]
            reasoning = f"Matched to {primary_provider['name']} using semantic search based on symptoms: '{symptoms}'."
            
            if priority:
                reasoning += f" Priority level: {priority}."
            
            if age and age < 18:
                reasoning += " Pediatric care recommended based on age."
        else:
            # Fallback if filtering removed all results
            primary_provider = potential_providers[0] if potential_providers else None
            reasoning = f"Matched to {primary_provider['name']} using semantic search."
        
        # Format matched providers list (top 5)
        matched_providers_list = [
            {
                "type": p["type"],
                "name": p["name"],
                "availability": p["availability"],
                "appointment_required": p["appointment_required"],
                "match_score": round(1.0 / (1.0 + p.get("match_score", 1.0)), 3)  # Convert distance to similarity score
            }
            for p in age_filtered[:5]
        ]
        
        return {
            "matched_providers": matched_providers_list,
            "primary_provider": {
                "type": primary_provider["type"],
                "name": primary_provider["name"],
                "availability": primary_provider["availability"],
                "appointment_required": primary_provider["appointment_required"]
            },
            "reasoning": reasoning,
            "symptoms": symptoms,
            "priority": priority,
            "context": context
        }
    
    except Exception as e:
        # Fallback if RAG search fails
        print(f"Warning: RAG search failed: {e}")
        return _fallback_match(symptoms, priority, patient_info)


def _fallback_match(
    symptoms: str,
    priority: Optional[str],
    patient_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Fallback matching logic when RAG is unavailable."""
    # Default recommendation based on priority
    if priority == "emergency":
        primary_provider = PROVIDER_TYPES["emergency_room"]
        primary_provider["type"] = "emergency_room"
        reasoning = "Emergency condition - recommend Emergency Room."
    elif priority == "urgent":
        primary_provider = PROVIDER_TYPES["urgent_care"]
        primary_provider["type"] = "urgent_care"
        reasoning = "Urgent condition - recommend Urgent Care."
    else:
        primary_provider = PROVIDER_TYPES["primary_care"]
        primary_provider["type"] = "primary_care"
        reasoning = "Routine condition - recommend Primary Care Physician."
    
    return {
        "matched_providers": [{
            "type": primary_provider["type"],
            "name": primary_provider["name"],
            "availability": primary_provider["availability"],
            "appointment_required": primary_provider["appointment_required"],
            "match_score": 1.0
        }],
        "primary_provider": {
            "type": primary_provider["type"],
            "name": primary_provider["name"],
            "availability": primary_provider["availability"],
            "appointment_required": primary_provider["appointment_required"]
        },
        "reasoning": reasoning + " (Fallback mode - RAG unavailable)",
        "symptoms": symptoms,
        "priority": priority,
        "context": patient_info
    }
