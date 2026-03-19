"""Circuit breaker -- stop routing to consistently failing nerves."""
import json
import logging
import time
from arqitect.brain.config import r

logger = logging.getLogger(__name__)

CB_KEY = "synapse:circuit_breakers"
FAILURE_THRESHOLD = 3  # consecutive failures before opening
COOLDOWN_SECONDS = 300  # 5 minutes before half-open

# States: closed (normal), open (blocked), half_open (probe)


def record_success(nerve_name: str):
    """Record a successful invocation -- reset failure count, close circuit."""
    state = _get_state(nerve_name)
    state["failures"] = 0
    state["state"] = "closed"
    state["last_success"] = time.time()
    _set_state(nerve_name, state)


def record_failure(nerve_name: str):
    """Record a failure -- may open the circuit."""
    state = _get_state(nerve_name)
    state["failures"] = state.get("failures", 0) + 1
    state["last_failure"] = time.time()

    if state["failures"] >= FAILURE_THRESHOLD:
        state["state"] = "open"
        state["opened_at"] = time.time()
        logger.warning("[CIRCUIT] Opened circuit for '%s' after %d failures", nerve_name, state['failures'])

    _set_state(nerve_name, state)


def is_available(nerve_name: str) -> bool:
    """Check if a nerve is available for routing."""
    state = _get_state(nerve_name)
    circuit_state = state.get("state", "closed")

    if circuit_state == "closed":
        return True

    if circuit_state == "open":
        # Check if cooldown has passed
        opened_at = state.get("opened_at", 0)
        if time.time() - opened_at > COOLDOWN_SECONDS:
            state["state"] = "half_open"
            _set_state(nerve_name, state)
            logger.info("[CIRCUIT] Half-open: allowing probe for '%s'", nerve_name)
            return True
        return False

    if circuit_state == "half_open":
        return True  # Allow one probe

    return True


def get_all_states() -> dict[str, dict]:
    """Get circuit breaker states for all tracked nerves."""
    raw = r.hgetall(CB_KEY)
    return {k: json.loads(v) for k, v in raw.items()}


def _get_state(nerve_name: str) -> dict:
    raw = r.hget(CB_KEY, nerve_name)
    if raw:
        return json.loads(raw)
    return {"state": "closed", "failures": 0}


def _set_state(nerve_name: str, state: dict):
    r.hset(CB_KEY, nerve_name, json.dumps(state))
