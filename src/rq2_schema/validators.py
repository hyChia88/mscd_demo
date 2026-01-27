"""
Validators for RQ2 submission JSON.

Two validation layers:
1. JSON Schema validation (structure, types, enums)
2. Domain validation (storey exists in model, GUID exists in model)
"""

from typing import Any, Dict, List, Tuple

from jsonschema import Draft202012Validator


def compute_required_fill_rate(submission: Dict[str, Any]) -> float:
    """
    Compute fill rate for required "headline" leaf fields.

    This metric is thesis-friendly: counts only key required fields,
    keeping the metric stable even if schema evolves.

    Args:
        submission: The submission dict to check

    Returns:
        Fill rate between 0.0 and 1.0
    """
    required_paths = [
        ("issue", "issue_type"),
        ("issue", "severity"),
        ("issue", "issue_summary"),
        ("bim_reference", "element_guid"),
        ("bim_reference", "ifc_class"),
        ("bim_reference", "storey_name"),
    ]

    filled = 0
    for path in required_paths:
        cur: Any = submission
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and cur not in [None, "", [], {}, "unknown", "UNKNOWN", "unspecified"]:
            filled += 1

    return filled / len(required_paths)


def jsonschema_validate(
    schema: Dict[str, Any], submission: Dict[str, Any]
) -> List[str]:
    """
    Validate submission against JSON Schema.

    Args:
        schema: JSON Schema dict
        submission: Submission dict to validate

    Returns:
        List of error messages (empty if valid)
    """
    v = Draft202012Validator(schema)
    errors = []
    for e in sorted(v.iter_errors(submission), key=str):
        errors.append(e.message)
    return errors


def domain_validate(
    submission: Dict[str, Any],
    available_storeys: List[str],
    element_exists: bool,
) -> List[str]:
    """
    Validate submission against domain constraints.

    Checks:
    - storey_name exists in the IFC model
    - element_guid exists in the IFC model

    Args:
        submission: Submission dict to validate
        available_storeys: List of valid storey names from model
        element_exists: Whether the element GUID was found in model

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    storey = submission.get("bim_reference", {}).get("storey_name", "")
    if storey and available_storeys and storey not in available_storeys:
        errors.append(f"storey_name not in model: {storey}")

    guid = submission.get("bim_reference", {}).get("element_guid", "")
    if guid and not element_exists:
        errors.append(f"element_guid not found in model: {guid}")

    return errors


def validate_all(
    schema: Dict[str, Any],
    submission: Dict[str, Any],
    available_storeys: List[str],
    element_exists: bool,
) -> Tuple[bool, float, List[str]]:
    """
    Run all validations: JSON Schema + domain checks.

    Args:
        schema: JSON Schema dict
        submission: Submission dict to validate
        available_storeys: List of valid storey names from model
        element_exists: Whether the element GUID was found in model

    Returns:
        Tuple of (passed, fill_rate, errors)
        - passed: True if all validations pass
        - fill_rate: Required field fill rate (0.0 to 1.0)
        - errors: List of all error messages
    """
    fill_rate = compute_required_fill_rate(submission)

    errors = []
    errors.extend(jsonschema_validate(schema, submission))
    errors.extend(domain_validate(submission, available_storeys, element_exists))

    # Pass if fill_rate is sufficient AND no structural/domain errors
    # Allow fill_rate >= 0.8 to pass (5/6 fields) for flexibility
    passed = (fill_rate >= 0.8) and (len(errors) == 0)

    return passed, fill_rate, errors
