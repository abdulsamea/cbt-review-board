import base64
from typing import Mapping


def make_json_safe(obj):
    """
    Recursively convert ANY object into JSON-safe structures.
    Handles bytes, dict keys, tuples, sets, and unknown objects.
    """

    # --- bytes (ANYWHERE) ---
    if isinstance(obj, (bytes, bytearray)):
        return {
            "__type__": "bytes",
            "base64": base64.b64encode(obj).decode("ascii"),
        }

    # --- dict / mapping ---
    if isinstance(obj, Mapping):
        safe_dict = {}
        for k, v in obj.items():
            # Keys can ALSO be bytes
            safe_key = make_json_safe(k)
            # JSON keys must be strings
            if not isinstance(safe_key, str):
                safe_key = str(safe_key)
            safe_dict[safe_key] = make_json_safe(v)
        return safe_dict

    # --- list / tuple / set ---
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(item) for item in obj]

    # --- primitives ---
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # --- fallback for unknown objects ---
    return str(obj)
