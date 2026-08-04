class GraphRepository:
    """Local fallback graph repository used when Neo4j is unavailable."""

    def __init__(self) -> None:
        self._graphs: dict[str, dict] = {}

    def save(self, case_id: str, graph: dict) -> dict:
        self._graphs[case_id] = graph
        return graph

    def get(self, case_id: str) -> dict | None:
        return self._graphs.get(case_id)
