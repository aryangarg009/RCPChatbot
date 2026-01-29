# metrics.py
from typing import Dict

# Canonical metric explanations keyed by your CSV column names
METRIC_EXPLANATIONS: Dict[str, str] = {
    "average_sparc": (
        "SPARC measures movement smoothness. Values closer to 0 usually indicate smoother, "
        "more coordinated movement. More negative values suggest jerkier movement."
    ),
    "area": (
        "Area is used here as a proxy for range of motion during the task. Higher values "
        "generally suggest a larger range of motion."
    ),
    "avg_efficiency": (
        "Efficiency reflects how effectively the patient completes the movement. Higher values "
        "generally suggest more efficient, accurate movement."
    ),
    "f_patient": (
        "Patient-applied force reflects how much force the patient is applying during the task. "
        "Higher values generally suggest greater force output."
    ),
    # Add more over time...
}

# Optional: human-friendly display names for narration
METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "average_sparc": "smoothness (SPARC)",
    "area": "range of motion (area)",
    "avg_efficiency": "efficiency",
    "f_patient": "total force",
}
