"""Config loader: reads config/experiment.yaml and resolves per-condition overrides.

Works in plain CPython, Blender's Python and Isaac Sim's Python. PyYAML is used when
available; otherwise a small fallback parser handles the subset of YAML this project uses.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fallback_yaml_load(text: str) -> Any:
    """Minimal YAML subset: nested maps by indentation, flow lists/maps, scalars, comments."""
    import ast

    def strip_comment(line: str) -> str:
        out, in_q = [], None
        for ch in line:
            if in_q:
                out.append(ch)
                if ch == in_q:
                    in_q = None
            elif ch in "'\"":
                in_q = ch
                out.append(ch)
            elif ch == "#":
                break
            else:
                out.append(ch)
        return "".join(out).rstrip()

    def scalar(tok: str) -> Any:
        t = tok.strip()
        if t == "" or t in ("null", "~", "None"):
            return None
        if t in ("true", "True"):
            return True
        if t in ("false", "False"):
            return False
        if t[0] in "[{":
            # flow collection: convert bare words to quoted strings then literal_eval
            import re

            def quote(m):
                w = m.group(0)
                if w in ("true", "false", "null", "True", "False", "None"):
                    return {"true": "True", "false": "False", "null": "None"}.get(w, w)
                try:
                    float(w)
                    return w
                except ValueError:
                    return repr(w)

            fixed = re.sub(r"(?<![\w'\"./-])[A-Za-z_][\w./-]*(?![\w'\"./-])", quote, t)
            fixed = fixed.replace("{", "{").replace(":", ":")
            # keys in flow maps: a: b -> 'a': b
            fixed = re.sub(r"'([A-Za-z_][\w]*)'\s*:", r"'\1':", fixed)
            return ast.literal_eval(fixed)
        if t[0] in "'\"":
            return t[1:-1]
        try:
            return int(t)
        except ValueError:
            pass
        try:
            return float(t)
        except ValueError:
            return t

    raw = [strip_comment(l) for l in text.splitlines()]
    # join continuation lines of multi-line flow collections ([...] / {...})
    lines, buf, depth = [], "", 0
    for l in raw:
        if not l.strip() and depth == 0:
            continue
        buf = (buf + " " + l.strip()) if depth > 0 else l
        depth += l.count("[") + l.count("{") - l.count("]") - l.count("}")
        if depth <= 0:
            lines.append(buf)
            buf, depth = "", 0
    if buf:
        lines.append(buf)

    def parse_block(i: int, indent: int):
        result: dict = {}
        while i < len(lines):
            line = lines[i]
            cur = len(line) - len(line.lstrip())
            if cur < indent:
                break
            if cur > indent:
                raise ValueError(f"bad indent at line: {line!r}")
            body = line.strip()
            if body.startswith("- "):
                raise ValueError("block lists are not supported by the fallback parser; use flow lists")
            key, _, val = body.partition(":")
            key = key.strip()
            if val.strip() == "":
                # nested block
                j = i + 1
                if j < len(lines) and (len(lines[j]) - len(lines[j].lstrip())) > indent:
                    sub, i = parse_block(j, len(lines[j]) - len(lines[j].lstrip()))
                    result[key] = sub
                    continue
                result[key] = None
                i += 1
            else:
                result[key] = scalar(val)
                i += 1
        return result, i

    data, _ = parse_block(0, 0)
    return data


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return _fallback_yaml_load(text)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_condition(cfg: dict) -> str:
    conds = cfg.get("conditions", {})
    for name, spec in conds.items():
        if spec and spec.get("default"):
            return name
    if conds:
        return next(iter(conds))
    raise KeyError("no conditions defined in config")


def resolve_condition(cfg: dict, condition: str | None) -> dict:
    """Return a full config with the named condition's overrides merged in.

    Adds `cfg['condition']` = {name, control}. Keys `default`/`control` in the condition
    spec are metadata, not overrides.
    """
    name = condition or default_condition(cfg)
    conds = cfg.get("conditions", {})
    if name not in conds:
        raise KeyError(f"unknown condition {name!r}; known: {sorted(conds)}")
    spec = dict(conds[name] or {})
    meta = {"name": name, "control": bool(spec.pop("control", False)), "default": bool(spec.pop("default", False))}
    merged = deep_merge({k: v for k, v in cfg.items() if k != "conditions"}, spec)
    merged["condition"] = meta
    return merged


def load_config(path: str, condition: str | None = None) -> dict:
    cfg = load_yaml(path)
    cfg["_config_path"] = os.path.abspath(path)
    if condition is None:
        return cfg
    return resolve_condition(cfg, condition)


def config_json(cfg: dict) -> str:
    return json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=2, sort_keys=True)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "config", "experiment.yaml")
    c = sys.argv[2] if len(sys.argv) > 2 else None
    print(config_json(load_config(p, c)))
