"""
E-T-A RAG Prototype - Role-Based Access Control
Mirrors the RBAC matrix from the architecture design (Section 7).

Key principle: Filter at retrieval, never post-process.
Chunks the user cannot access are never retrieved — they never reach the LLM.
"""

# ─── Access Level Definitions ────────────────────────────────────────
# Each document chunk is tagged with one of these access levels
ACCESS_LEVELS = {
    "public": "Accessible by all authenticated employees",
    "department": "Accessible only by members of the owning department",
    "restricted": "Accessible only by specific roles (HR, Management)",
    "confidential": "Management only",
}

# ─── Role → Accessible Data Categories ──────────────────────────────
# Maps to the RBAC table in architecture design Section 7
ROLE_ACCESS = {
    "all_staff": {
        "allowed_access_levels": ["public"],
        "allowed_departments": None,  # No department-level access
        "description": "Basic employee access - product manuals, public docs",
        "questions_can_ask": ["What are E-T-A's core values?", "How do I reset my password?", "What are the holiday policies?"],
        "questions_cannot_ask": ["What is the Q3 revenue?", "Show me the CAD models for the new circuit breaker", "What is David Braun's salary?"]
    },
    "engineering": {
        "allowed_access_levels": ["public", "department"],
        "allowed_departments": ["engineering"],
        "description": "Engineering specs, test reports, CAD metadata, BOMs",
        "questions_can_ask": ["What are the specifications for the 3120-N series?", "How is testing conducted for thermal circuit breakers?"],
        "questions_cannot_ask": ["Show me employee performance reviews.", "What is the marketing budget?"]
    },
    "production": {
        "allowed_access_levels": ["public", "department"],
        "allowed_departments": ["production", "engineering"],  # Production can see engineering specs
        "description": "Production data, SOPs, work instructions, CAD files",
        "questions_can_ask": ["What is the assembly process for 1140 series?", "Are there any known defects reported for relay X?"],
        "questions_cannot_ask": ["What is the company's financial forecast?", "Can I see the HR onboarding manual?"]
    },
    "quality": {
        "allowed_access_levels": ["public", "department"],
        "allowed_departments": ["quality"],
        "description": "Audit checklists, certification reports, standards",
        "questions_can_ask": ["What are the ISO 9001 compliance requirements?", "Show me the latest audit checklists."],
        "questions_cannot_ask": ["What is the R&D budget for next year?", "What is the HR policy for remote work?"]
    },
    "hr": {
        "allowed_access_levels": ["public", "department", "restricted"],
        "allowed_departments": ["hr"],
        "description": "HR records, employee data, policies",
        "questions_can_ask": ["What are the remote work guidelines?", "How do I onboard a new employee?", "What is the standard vacation allowance?"],
        "questions_cannot_ask": ["What are the technical specs of the 2210 series?", "Show me the production schedule for next month."]
    },
    "finance": {
        "allowed_access_levels": ["public", "department"],
        "allowed_departments": ["finance"],
        "description": "Budget reports, FI/CO data",
        "questions_can_ask": ["What was the Q2 operating expense?", "How do I submit an expense report?"],
        "questions_cannot_ask": ["What are the design tolerances for part number 1234?", "Show me the production line SOP."]
    },
    "management": {
        "allowed_access_levels": ["public", "department", "restricted", "confidential"],
        "allowed_departments": None,  # Access to all departments
        "description": "Full access to all data categories",
        "questions_can_ask": ["What is the overall Q3 financial performance?", "Are there any critical quality issues reported?", "What is the status of the new HR policy?"],
        "questions_cannot_ask": ["(Can ask about anything across departments)"]
    },
}

# ─── Simulated Users (for prototype testing) ─────────────────────────
MOCK_USERS = {
    "thomas.mueller": {
        "name": "Thomas Mueller",
        "role": "engineering",
        "department": "engineering",
    },
    "claudia.schmidt": {
        "name": "Claudia Schmidt",
        "role": "finance",
        "department": "finance",
    },
    "david.braun": {
        "name": "David Braun",
        "role": "management",
        "department": "operations",
    },
    "anna.weber": {
        "name": "Anna Weber",
        "role": "hr",
        "department": "hr",
    },
    "max.fischer": {
        "name": "Max Fischer",
        "role": "all_staff",
        "department": "sales",
    },
    "alexandru.pop": {
        "name": "Alexandru Pop",
        "role": "production",
        "department": "production",
    },
}


def get_retrieval_filter(user_role: str) -> dict:
    """
    Build a metadata filter for the vector store based on user role.
    This implements pre-retrieval RBAC filtering.
    
    Returns a filter dict compatible with ChromaDB's where clause.
    In production, this would be a Qdrant filter.
    """
    role_config = ROLE_ACCESS.get(user_role, ROLE_ACCESS["all_staff"])
    allowed_levels = role_config["allowed_access_levels"]
    allowed_depts = role_config["allowed_departments"]

    # Build filter conditions
    conditions = []

    # Access level filter
    conditions.append({
        "access_level": {"$in": allowed_levels}
    })

    # Department filter (if not management with full access)
    if allowed_depts is not None:
        conditions.append({
            "$or": [
                {"access_level": {"$eq": "public"}},
                {"department": {"$in": allowed_depts}},
            ]
        })

    # Combine conditions
    if len(conditions) == 1:
        return {"$and": [conditions[0]]}
    return {"$and": conditions}


def can_access(user_role: str, chunk_metadata: dict) -> bool:
    """
    Check if a user role can access a specific chunk.
    Used as a fallback / validation check.
    """
    role_config = ROLE_ACCESS.get(user_role, ROLE_ACCESS["all_staff"])
    allowed_levels = role_config["allowed_access_levels"]
    allowed_depts = role_config["allowed_departments"]

    chunk_level = chunk_metadata.get("access_level", "confidential")
    chunk_dept = chunk_metadata.get("department", "unknown")

    # Check access level
    if chunk_level not in allowed_levels:
        return False

    # Check department (management has access to all)
    if allowed_depts is not None and chunk_level != "public":
        if chunk_dept not in allowed_depts:
            return False

    return True
