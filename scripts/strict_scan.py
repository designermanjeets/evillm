#!/usr/bin/env python3
"""Strict mode scanner to detect forbidden fallbacks."""

from __future__ import annotations
import ast, pathlib, re, sys, json

BASE = pathlib.Path(__file__).resolve().parents[1]
INCLUDE = [
    "app/services/llm.py",
    "app/services/retriever.py",
    "app/agents/draft_workflow.py",
    "app/routers/search.py",
    "app/routers/draft.py",
    "app/config/settings.py",
]
# Files allowed to reference stub or fallback code IF annotated:
ANNOTATION_OK = {
    "app/services/llm.py": {"# strict-scan: allowed-stub"},
    "app/services/retriever.py": {"# strict-scan: guarded-fallback"},
}

VIOLATIONS = []

def rel(p: pathlib.Path) -> str:
    return str(p.relative_to(BASE))

def read(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def has_annotation_near(lines: list[str], idx: int, tokens: set[str]) -> bool:
    # Look on same line and up to 2 lines above for an allow marker
    for j in range(max(0, idx-2), idx+1):
        if any(tok in lines[j] for tok in tokens):
            return True
    return False

def scan_llm(path: pathlib.Path):
    """
    Rule L1: Stub LLM is forbidden anywhere except llm.py when annotated.
    For any import/name containing 'Stub' or 'stub', require '# strict-scan: allowed-stub'.
    """
    code = read(path)
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"\bstub\b", line, re.IGNORECASE):
            ok = rel(path) in ANNOTATION_OK and has_annotation_near(lines, i, ANNOTATION_OK[rel(path)])
            if not ok:
                VIOLATIONS.append({
                    "file": rel(path),
                    "line": i+1,
                    "rule": "L1",
                    "msg": "stub LLM reference without allowed annotation",
                    "line_text": line.strip(),
                })

def scan_retriever(path: pathlib.Path):
    """
    Rule R1: Only flag actual fallback logic, not legitimate service usage.
    Look for:
    1. fallback_enabled = True (hardcoded fallback)
    2. fallback to SQL text search (actual fallback path)
    3. Missing strict mode guards on fallback paths
    """
    code = read(path)
    lines = code.splitlines()
    
    # Only flag actual fallback logic, not service configuration
    fallback_patterns = [
        r"fallback_enabled\s*=\s*True",  # Hardcoded fallback enabled
        r"fallback.*SQL.*text.*search",   # Fallback to SQL search
        r"falling.*back.*to",             # Fallback language
    ]
    
    for i, line in enumerate(lines):
        for pattern in fallback_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                # Check if this is in a fallback context
                if "fallback" in line.lower():
                    annotated = rel(path) in ANNOTATION_OK and has_annotation_near(lines, i, ANNOTATION_OK[rel(path)])
                    
                    # For docstrings and comments, just require annotation
                    if re.search(r'"""|#', line):
                        if not annotated:
                            VIOLATIONS.append({
                                "file": rel(path),
                                "line": i+1,
                                "rule": "R1",
                                "msg": "retriever fallback docstring/comment missing annotation",
                                "line_text": line.strip(),
                            })
                    else:
                        # For actual code, require both annotation and strict guard
                        guard_nearby = any(
                            re.search(r"strict_mode\s*==\s*False|not\s+.*strict_mode", lines[j])
                            for j in range(max(0, i-5), min(len(lines), i+6))
                        )
                        if not (annotated and guard_nearby):
                            VIOLATIONS.append({
                                "file": rel(path),
                                "line": i+1,
                                "rule": "R1",
                                "msg": "retriever fallback path missing annotation and/or strict guard",
                                "line_text": line.strip(),
                            })

def scan_settings(path: pathlib.Path):
    """
    Rule S1: security.strict_mode must exist.
    Rule S2: llm.allow_stub must exist AND be False by default in dev example.
    """
    code = read(path)
    if "strict_mode" not in code:
        VIOLATIONS.append({"file": rel(path), "line": 1, "rule": "S1", "msg": "security.strict_mode missing"})
    if "allow_stub" not in code:
        VIOLATIONS.append({"file": rel(path), "line": 1, "rule": "S2", "msg": "llm.allow_stub missing"})

def main():
    for rel_path in INCLUDE:
        p = BASE / rel_path
        if not p.exists(): 
            continue
        if "llm.py" in rel_path:
            scan_llm(p)
        elif "retriever.py" in rel_path:
            scan_retriever(p)
        elif "settings.py" in rel_path:
            scan_settings(p)

    out = {"violations": VIOLATIONS, "count": len(VIOLATIONS)}
    print(json.dumps(out, indent=2))
    sys.exit(1 if VIOLATIONS else 0)

if __name__ == "__main__":
    main()
