from app.core.security import utc_now_iso
from app.models.schemas import AuditEvent


class AuditService:
    def __init__(self) -> None:
        self._events: dict[str, list[AuditEvent]] = {}

    def record(
        self,
        case_id: str,
        event_type: str,
        message: str,
        actor: str = "system",
        metadata: dict | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            case_id=case_id,
            event_type=event_type,
            actor=actor,
            message=message,
            metadata=metadata or {},
            created_at=utc_now_iso(),
        )
        self._events.setdefault(case_id, []).append(event)
        return event

    def list_for_case(self, case_id: str) -> list[AuditEvent]:
        return list(self._events.get(case_id, []))
