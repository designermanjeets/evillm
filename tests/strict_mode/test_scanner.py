import json, subprocess, sys, pathlib

def test_strict_scan_clean():
    repo = pathlib.Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, "scripts/strict_scan.py"], cwd=repo, capture_output=True, text=True)
    assert proc.returncode == 0, f"strict-scan failed:\n{proc.stdout}\n{proc.stderr}"
    data = json.loads(proc.stdout or "{}")
    assert data.get("count", 0) == 0
