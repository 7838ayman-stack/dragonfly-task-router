"""Pydantic models for the Dragonfly dynamic task router (routing only)."""

from pydantic import BaseModel, Field


class ServerState(BaseModel):
    """Live telemetry snapshot for a single active server in the pool."""

    server_id: str = Field(
        description="Unique identifier of the server, e.g. 'server-node-3'.",
    )
    cpu_usage: float = Field(
        ge=0.0,
        le=100.0,
        description="Current CPU usage of the server, in percent (0-100).",
    )
    ram_usage: float = Field(
        ge=0.0,
        le=100.0,
        description="Current RAM usage of the server, in percent (0-100).",
    )
    active_connections: int = Field(
        ge=0,
        description="Number of connections currently handled by the server.",
    )


class IncomingTask(BaseModel):
    """Demand of one incoming task that must be routed to a server."""

    task_id: str = Field(
        description="Correlation id of the incoming task.",
    )
    required_cpu: float = Field(
        ge=0.0,
        description="CPU percent the task needs on its target server.",
    )


class RoutingResponse(BaseModel):
    """Routing decision produced by the Dragonfly task router."""

    selected_server_id: str = Field(
        description="Id of the server selected for the task.",
    )
    server_cpu_before: float = Field(
        description="CPU usage of the selected server before routing.",
    )
    server_cpu_after: float = Field(
        description="CPU usage of the selected server after the task was assigned.",
    )
    message: str = Field(
        description="Human-readable explanation of how the server was chosen.",
    )


class BatchRouteLog(BaseModel):
    """Audit trail entry for one task routed during a batch traffic simulation."""

    task_id: str = Field(
        description="Correlation id of the simulated task.",
    )
    task_type: str = Field(
        description="Human-readable workload category, e.g. 'DB Read'.",
    )
    required_cpu: float = Field(
        ge=0.0,
        description="CPU percent the task demanded on its target server.",
    )
    assigned_server: str = Field(
        description="Id of the server the router assigned the task to.",
    )
    cpu_before: float = Field(
        description="CPU usage of the assigned server before the task landed.",
    )
    cpu_after: float = Field(
        description="CPU usage of the assigned server after the task landed.",
    )
    latency_ms: float = Field(
        ge=0.0,
        description=(
            "Routing decision latency measured with time.perf_counter(), "
            "in milliseconds (3-decimal precision)."
        ),
    )
