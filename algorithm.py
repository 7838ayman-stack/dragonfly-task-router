"""Dragonfly Algorithm (DA) dynamic task router (intelligent load balancer).

Implements the metaheuristic of Mirjalili (2015), "Dragonfly algorithm: a new
meta-heuristic optimization technique ...", using the 5 classic operators:

    S (separation)        =  sum(X_j - X_i)          over neighbors
    A (alignment)         =  mean(step_j) - step_i   over neighbors
    C (cohesion)          =  mean(X_j) - X_i         over neighbors
    F (food attraction)   =  food_position - X_i
    E (enemy distraction) =  enemy_position + X_i

Step update:  step <- w * step + 0.1*S + 0.1*A + 0.5*C + 2.0*F + 2.0*E
              step <- clip(step, -1, 1)
Position:     X     <- clip(X + step, lower, upper)

The router indexes a VOLATILE server pool (1 to N servers, mutated by
``randomize_server_states()`` from another request thread). Every routing
call takes one atomic snapshot of the pool and evaluates the whole swarm
against it, so a concurrent pool swap can never produce an out-of-bounds
access or an index error.
"""

from collections.abc import Callable

import numpy as np
from fastapi import HTTPException

from dragonfly import Dragonfly
from schemas import IncomingTask, RoutingResponse, ServerState
from server_pool import SERVERS

# --- DA operator coefficients (s, a, c, f, e) -------------------------------
SEPARATION_COEF = 0.1
ALIGNMENT_COEF = 0.1
COHESION_COEF = 0.5
FOOD_COEF = 2.0
ENEMY_COEF = 2.0

# --- Inertia weight schedule: w(t) = 0.9 -> 0.4 over iterations ------------
INERTIA_START = 0.9
INERTIA_END = 0.4

# --- Neighborhood radius schedule: r(t) = 1.0 -> 0.0 in search space -------
NEIGHBORHOOD_RADIUS_START = 1.0

# --- Step vector bound (matches the Dragonfly.__init__ [-1.0, 1.0] range) --
STEP_BOUND = 1.0

# --- Routing hyper-parameters ----------------------------------------------
ROUTING_SWARM_SIZE = 20
ROUTING_MAX_ITER = 40
HEAVY_PENALTY = 10000.0


def _snapshot_servers() -> list[tuple[str, ServerState]]:
    """Atomically snapshot the current pool (safe under concurrent swaps)."""
    return list(SERVERS.items())


def _map_position_to_index(position: np.ndarray, pool_size: int) -> int:
    """Safely convert a continuous position to a discrete pool index."""
    return int(round(np.clip(float(position[0]), 0.0, float(pool_size - 1))))


def _evaluate_routing_fitness(
    position: np.ndarray,
    task: IncomingTask,
    snapshot: list[tuple[str, ServerState]],
) -> float:
    """Score the server mapped to ``position`` for ``task`` (maximization).

    The continuous 1D position is clipped and rounded to a discrete index
    against the snapshot length, then scored on free CPU and connection
    pressure. A massive penalty disqualifies overloaded servers.
    """
    if not snapshot:
        raise HTTPException(
            status_code=500,
            detail="Server pool is empty; cannot route task.",
        )

    index = _map_position_to_index(position, len(snapshot))
    server = snapshot[index][1]

    available_cpu = 100.0 - server.cpu_usage
    base_score = (available_cpu * 2.0) - (server.active_connections * 5.0)

    if server.cpu_usage > 90.0 or available_cpu < task.required_cpu:
        base_score -= HEAVY_PENALTY

    return float(base_score)


def _run_da(
    bounds: tuple,
    swarm_size: int,
    max_iter: int,
    fitness_fn: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Run the Mirjalili DA loop.

    Returns ``(food_pos, food_fitness, enemy_pos, enemy_fitness)`` where the
    food is the global best and the enemy the global worst position seen.
    """
    lower: float = float(bounds[0])
    upper: float = float(bounds[1])
    span: float = upper - lower

    swarm: list[Dragonfly] = [Dragonfly(bounds) for _ in range(swarm_size)]

    best_fitness = float("-inf")
    worst_fitness = float("inf")
    food_pos: np.ndarray = swarm[0].position.copy()
    enemy_pos: np.ndarray = swarm[0].position.copy()

    for iteration in range(max_iter):
        progress = iteration / max_iter
        # Dynamically decaying inertia weight: 0.9 -> 0.4.
        w = INERTIA_START - (INERTIA_START - INERTIA_END) * progress
        # Neighborhood radius decays linearly: 1.0 -> ~0.0.
        neighbourhood = NEIGHBORHOOD_RADIUS_START * (1.0 - progress) * span

        # 1) Evaluate fitness for every dragonfly.
        for dragonfly in swarm:
            dragonfly.fitness = fitness_fn(dragonfly.position)

        # 2) Update the global food (max fitness) and enemy (min fitness).
        for dragonfly in swarm:
            if dragonfly.fitness > best_fitness:
                best_fitness = float(dragonfly.fitness)
                food_pos = dragonfly.position.copy()
            if dragonfly.fitness < worst_fitness:
                worst_fitness = float(dragonfly.fitness)
                enemy_pos = dragonfly.position.copy()

        # Snapshot current state so all operators use consistent values.
        positions = np.array([dragonfly.position for dragonfly in swarm])
        steps = np.array([dragonfly.step for dragonfly in swarm])
        new_positions = np.empty_like(positions)
        new_steps = np.empty_like(steps)

        # 3) Radius-based neighborhoods (S/A/C are neighbor-only operators).
        for i in range(swarm_size):
            neighbors = [
                j
                for j in range(swarm_size)
                if j != i
                and abs(float(positions[j, 0]) - float(positions[i, 0]))
                <= neighbourhood
            ]

            if neighbors:
                neighbor_positions = positions[neighbors]
                neighbor_steps = steps[neighbors]
                # Mirjalili's exact operator math.
                separation = np.sum(neighbor_positions - positions[i], axis=0)
                alignment = np.mean(neighbor_steps, axis=0) - steps[i]
                cohesion = np.mean(neighbor_positions, axis=0) - positions[i]
            else:
                # Isolated dragonfly: zeroed operators keep the update defined.
                separation = np.zeros(1)
                alignment = np.zeros(1)
                cohesion = np.zeros(1)

            food_attraction = food_pos - positions[i]
            enemy_distraction = enemy_pos + positions[i]

            updated_step = (
                w * steps[i]
                + SEPARATION_COEF * separation
                + ALIGNMENT_COEF * alignment
                + COHESION_COEF * cohesion
                + FOOD_COEF * food_attraction
                + ENEMY_COEF * enemy_distraction
            )
            new_steps[i] = np.clip(updated_step, -STEP_BOUND, STEP_BOUND)
            new_positions[i] = positions[i] + new_steps[i]

        # 4) Apply movement with STRICT bounds enforcement.
        for i, dragonfly in enumerate(swarm):
            dragonfly.step = new_steps[i]
            dragonfly.position = np.clip(new_positions[i], lower, upper)

    return (
        food_pos,
        float(best_fitness),
        enemy_pos,
        float(worst_fitness),
    )


def _commit_task_to_server(server: ServerState, task: IncomingTask) -> None:
    """Commit a routing decision by mutating the shared server object in place.

    ``server`` is the very same ``ServerState`` instance stored inside the
    global ``SERVERS`` dictionary, so these writes are instantly visible to
    every subsequent routing call and to the ``/servers`` endpoint. This is
    the stateful fix for the thundering-herd problem: the next swarm now
    scores this server as more loaded and naturally spreads the burst.
    """
    server.cpu_usage = min(100.0, server.cpu_usage + task.required_cpu)
    server.active_connections += 1


def run_routing_optimization(task: IncomingTask) -> RoutingResponse:
    """Route a task to the best server in the (dynamic) pool via the DA."""
    snapshot = _snapshot_servers()

    # Edge Case Handler 1: nothing to route to.
    if len(snapshot) == 0:
        raise HTTPException(
            status_code=500,
            detail="Server pool is empty; cannot route task.",
        )

    # Edge Case Handler 2: single server -> O(1) bypass, no DA cycles burned.
    if len(snapshot) == 1:
        only_id, only_server = snapshot[0]
        live_server = SERVERS.get(only_id, only_server)
        cpu_before = float(live_server.cpu_usage)
        _commit_task_to_server(live_server, task)
        return RoutingResponse(
            selected_server_id=only_id,
            server_cpu_before=cpu_before,
            server_cpu_after=float(live_server.cpu_usage),
            message=(
                "Bypassed the Dragonfly Algorithm: only one server is active, "
                "so it was selected in O(1) time."
            ),
        )

    # Search space maps a continuous position onto the pool index range.
    bounds: tuple = (0.0, float(len(snapshot) - 1))

    food_pos, food_fitness, _, _ = _run_da(
        bounds,
        ROUTING_SWARM_SIZE,
        ROUTING_MAX_ITER,
        lambda position: _evaluate_routing_fitness(position, task, snapshot),
    )

    # Final mapping: re-snapshot (pool may have shifted mid-run) and re-clip.
    final_snapshot = _snapshot_servers()
    if not final_snapshot:
        raise HTTPException(
            status_code=500,
            detail="Server pool became empty during routing.",
        )

    best_index = _map_position_to_index(food_pos, len(final_snapshot))
    best_id, best_server = final_snapshot[best_index]

    # Stateful commitment: re-fetch the LIVE object from the global pool
    # (the pool may have been swapped mid-run) and mutate it in place so
    # the next request already sees this server as occupied.
    live_server = SERVERS.get(best_id, best_server)
    cpu_before = float(live_server.cpu_usage)
    _commit_task_to_server(live_server, task)

    return RoutingResponse(
        selected_server_id=str(best_id),
        server_cpu_before=cpu_before,
        server_cpu_after=float(live_server.cpu_usage),
        message=(
            f"Routed task '{task.task_id}' to '{best_id}' via the Dragonfly "
            f"Algorithm ({ROUTING_SWARM_SIZE} agents x {ROUTING_MAX_ITER} "
            f"iterations, best fitness {food_fitness:.2f})."
        ),
    )
