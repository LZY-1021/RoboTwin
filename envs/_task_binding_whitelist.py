import json
import os


ENV_NAME = "ROBOTWIN_ALLOWED_TASK_BINDINGS_JSON"


def load_task_binding_whitelist(expected_task, environ=None):
    """Load the collector-injected role whitelist for one task.

    An empty environment variable keeps native RoboTwin behavior.  Once the
    collector injects a policy, malformed or cross-task payloads fail closed.
    """
    source = os.environ if environ is None else environ
    raw = str(source.get(ENV_NAME, "")).strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        task = payload["task"]
        fields = payload["fields"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid {ENV_NAME} payload") from exc
    if task != expected_task:
        raise ValueError(
            f"{ENV_NAME} task mismatch: expected {expected_task}, got {task}"
        )
    if not isinstance(fields, dict) or not fields:
        raise ValueError(f"{ENV_NAME} fields must be a non-empty object")

    normalized = {}
    for role, values in fields.items():
        if not isinstance(role, str) or not isinstance(values, list) or not values:
            raise ValueError(f"invalid binding whitelist for role {role!r}")
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"invalid binding value for role {role!r}")
        normalized[role] = tuple(dict.fromkeys(values))
    return normalized


def parse_model_binding(binding, object_families, available_ids_for):
    """Validate and split a ``model/baseN`` binding."""
    if not isinstance(binding, str) or "/base" not in binding:
        raise ValueError(f"model binding must use model/baseN format: {binding!r}")
    modelname, model_id_text = binding.rsplit("/base", 1)
    try:
        model_id = int(model_id_text)
    except ValueError as exc:
        raise ValueError(f"model binding has invalid base id: {binding!r}") from exc
    if modelname not in object_families:
        raise ValueError(f"model binding is outside native families: {binding!r}")
    if model_id not in available_ids_for(modelname):
        raise ValueError(f"model binding is outside native assets: {binding!r}")
    return modelname, model_id


def build_distinct_model_binding_pairs(
    fields,
    role_a,
    role_b,
    object_families,
    available_ids_for,
):
    """Build the legal Cartesian product for two distinct object roles."""
    if not fields:
        return []
    try:
        raw_a = fields[role_a]
        raw_b = fields[role_b]
    except KeyError as exc:
        raise ValueError(f"missing binding role {exc.args[0]!r}") from exc
    a_choices = [
        parse_model_binding(value, object_families, available_ids_for)
        for value in raw_a
    ]
    b_choices = [
        parse_model_binding(value, object_families, available_ids_for)
        for value in raw_b
    ]
    pairs = [
        (a_name, a_id, b_name, b_id)
        for a_name, a_id in a_choices
        for b_name, b_id in b_choices
        if a_name != b_name
    ]
    if not pairs:
        raise ValueError(
            f"binding roles {role_a!r}/{role_b!r} have no distinct legal pairs"
        )
    return pairs
