"""Small shared UI helpers."""


def status_icon(status: str) -> str:
    """Return the dashboard's consistent colored status icon."""

    colors = {
        "EXCELLENT": "🟢",
        "GOOD": "🔵",
        "FAIR": "🟠",
        "POOR": "🔴",
        "HIGH": "🟢",
        "MEDIUM": "🟠",
        "LOW": "🔴"
    }

    return colors.get(status, "⚪")
