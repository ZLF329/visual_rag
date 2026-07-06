from __future__ import annotations


def build_agent_human_turn(system: str, user: str, call_type: str | None = None) -> str:
    parts: list[str] = []
    if call_type:
        parts.append(f"Agent call type: {call_type}")

    system = system.strip()
    if system:
        parts.append(f"Instruction:\n{system}")

    parts.append(f"Input:\n{user.strip()}")
    return "\n\n".join(part for part in parts if part)
