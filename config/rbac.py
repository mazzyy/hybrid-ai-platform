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
    },
    "engineering": {
        "allowed_access_levels": ["public", "department"],
        "allowed_departments": ["engineering"],
        "description": "Engineering specs, test reports, CAD metadata, BOMs",
    },
    "production": {
        "allowed_access_levels": ["public", "department"],
        "allowed_departments": ["production", "engineering"],  # Production can see engineering specs
        "description": "Production data, SOPs, work instructions, CAD files",
    },
    "quality": {
        "allowed_access_levels": ["public", "department"],
        "allowed_departments": ["quality"],
        "description": "Audit checklists, certification reports, standards",
    },
    "hr": {
        "allowed_access_levels": ["public", "department", "restricted"],
        "allowed_departments": ["hr"],
        "description": "HR records, employee data, policies",
    },
    "finance": {
        "allowed_access_levels": ["public", "department"],
        "allowed_departments": ["finance"],
        "description": "Budget reports, FI/CO data",
    },
    "management": {
        "allowed_access_levels": ["public", "department", "restricted", "confidential"],
        "allowed_departments": None,  # Access to all departments
        "description": "Full access to all data categories",
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
