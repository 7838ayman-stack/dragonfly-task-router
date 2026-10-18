"""FastAPI service: Dragonfly Algorithm dynamic task router (load balancer)."""

import random
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from algorithm import run_routing_optimization
from schemas import BatchRouteLog, IncomingTask, RoutingResponse, ServerState
from server_pool import SERVERS, randomize_server_states

app = FastAPI(
    title="Dragonfly Algorithm Dynamic Task Router",
    description=(
        "Intelligent load balancer powered by the Dragonfly Algorithm. Routes "
        "incoming tasks to the best server in a dynamic 3-15 server pool "
        "using Mirjalili's swarm metaheuristic."
    ),
    version="4.0.0",
)

# --- Batch traffic simulation parameters ------------------------------------
BATCH_SIMULATION_TASK_COUNT: int = 10

MOCK_WORKLOADS: list[dict[str, float | str]] = [
    {"type": "Small Query", "cpu": 5.0},
    {"type": "DB Read", "cpu": 15.0},
    {"type": "Git Commit", "cpu": 35.0},
    {"type": "Heavy ML Training", "cpu": 65.0},
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/route-task", response_model=RoutingResponse)
def route_task(task: IncomingTask) -> RoutingResponse:
    """Pick the best server for a task via the Dragonfly Algorithm."""
    try:
        return run_routing_optimization(task)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/simulate-batch", response_model=list[BatchRouteLog])
def simulate_batch() -> list[BatchRouteLog]:
    """Burst 10 mixed workloads at the router and log live load distribution.

    Each iteration draws a random workload template, times the stateful
    routing decision with microsecond precision, and records the server's
    CPU before/after the task was committed, proving that consecutive
    requests spread across the pool instead of piling onto one server.
    """
    logs: list[BatchRouteLog] = []

    for index in range(BATCH_SIMULATION_TASK_COUNT):
        workload = random.choice(MOCK_WORKLOADS)
        task = IncomingTask(
            task_id=f"batch-task-{index + 1}",
            required_cpu=float(workload["cpu"]),
        )

        start = time.perf_counter()
        routing = run_routing_optimization(task)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)

        logs.append(
            BatchRouteLog(
                task_id=task.task_id,
                task_type=str(workload["type"]),
                required_cpu=task.required_cpu,
                assigned_server=routing.selected_server_id,
                cpu_before=routing.server_cpu_before,
                cpu_after=routing.server_cpu_after,
                latency_ms=latency_ms,
            )
        )

    return logs


@app.get("/servers", response_model=dict[str, ServerState])
def list_servers() -> dict[str, ServerState]:
    """Return the current dynamic pool (shows the live cloud state)."""
    return dict(SERVERS)


@app.post("/randomize-servers", response_model=dict[str, ServerState])
def randomize_servers() -> dict[str, ServerState]:
    """Spin servers up/down: re-randomize pool size (3-15) and telemetry."""
    randomize_server_states()
    return dict(SERVERS)
