"""Global server pool state and dynamic mock data.

The pool size is DYNAMIC: every refresh spins between 3 and 100 servers up or
down (dict cleared and repopulated), simulating live cloud fluctuations.
All readers must treat the pool as volatile — never cache ``len(SERVERS)``
across a statement boundary without taking an atomic snapshot.
"""

import random

from schemas import ServerState

# Bhai, yahan change kiya hai: Min 3 aur Max 100
MIN_ACTIVE_SERVERS: int = 3
MAX_ACTIVE_SERVERS: int = 100

MAX_ACTIVE_CONNECTIONS: int = 500

SERVERS: dict[str, ServerState] = {}


def randomize_server_states() -> None:
    """Repopulate the pool with a random number of servers (3 to 100).

    Clears the shared dictionary first, then refills it with dynamically
    named servers ('server-node-1', ...) carrying random valid telemetry.
    """
    new_servers: dict[str, ServerState] = {}
    
    # Ye line ab 3 se 100 ke beech koi bhi random number of servers banayegi
    for index in range(1, random.randint(MIN_ACTIVE_SERVERS, MAX_ACTIVE_SERVERS) + 1):
        server_id = f"server-node-{index}"
        new_servers[server_id] = ServerState(
            server_id=server_id,
            cpu_usage=round(random.uniform(0.0, 100.0), 2),
            ram_usage=round(random.uniform(0.0, 100.0), 2),
            active_connections=random.randint(0, MAX_ACTIVE_CONNECTIONS),
        )

    SERVERS.clear()
    SERVERS.update(new_servers)


# Initialize the pool as soon as the service imports this module.
randomize_server_states()