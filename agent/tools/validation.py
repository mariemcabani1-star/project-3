"""Tiny strict JSON-Schema validator (stdlib only).

Supports: type, required, properties, additionalProperties:false, enum, pattern,
minLength/maxLength, minimum/maximum, items, minItems/maxItems.
Returns human-readable problems the model can act on.
"""
from __future__ import annotations

import re

_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate(value, schema: dict, path: str = "arguments") -> list[str]:
    problems: list[str] = []
    t = schema.get("type")
    if t:
        py = _TYPES[t]
        ok = isinstance(value, py) and not (t in ("integer", "number") and isinstance(value, bool))
        if not ok:
            return [f"'{path}' must be of type {t}, got {type(value).__name__}"]

    if "enum" in schema and value not in schema["enum"]:
        problems.append(f"'{path}' must be one of {schema['enum']}, got {value!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"'{path}' is too short (min {schema['minLength']} characters)")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            problems.append(f"'{path}' is too long (max {schema['maxLength']} characters)")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            hint = schema.get("x-format-hint", f"pattern {schema['pattern']}")
            problems.append(f"'{path}' has the wrong format; expected {hint}, got {value!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"'{path}' must be >= {schema['minimum']}, got {value}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"'{path}' must be <= {schema['maximum']}, got {value}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"'{path}' needs at least {schema['minItems']} item(s)")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"'{path}' allows at most {schema['maxItems']} item(s)")
        if "items" in schema:
            for i, item in enumerate(value):
                problems += validate(item, schema["items"], f"{path}[{i}]")

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                problems.append(f"missing required argument '{req}'")
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    problems.append(f"unknown argument '{k}' (allowed: {sorted(props)})")
        for k, sub in props.items():
            if k in value:
                problems += validate(value[k], sub, k if path == "arguments" else f"{path}.{k}")
    return problems
