from __future__ import annotations


_RISK_MESSAGES: dict[int, str] = {
    0: "Very low risk. Safe conditions.",
    1: "Low risk. Drive normally but stay cautious.",
    2: "Moderate risk. Stay alert and maintain lane discipline.",
    3: "High risk detected. Reduce speed and increase following distance.",
    4: "Critical risk ahead. Slow down immediately and create distance.",
}


def risk_to_message(level: int) -> str:
    try:
        return _RISK_MESSAGES[int(level)]
    except (ValueError, TypeError, KeyError):
        return "Unknown risk level."
