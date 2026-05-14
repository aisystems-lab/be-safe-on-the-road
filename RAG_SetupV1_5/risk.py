def risk_to_message(level: int):
    return {
        4: "Critical risk ahead! Slow down immediately and create distance.",
        3: "High risk detected. Reduce speed and increase following distance.",
        2: "Moderate risk. Stay alert and maintain lane discipline.",
        1: "Low risk. Drive normally but remain cautious.",
        0: "Very low risk. Safe conditions."
    }.get(level, "Unknown risk level")