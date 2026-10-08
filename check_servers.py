#!/usr/bin/env python3
"""
Pre-flight vLLM Server Health Check
-----------------------------------
Pings every vLLM server defined in config.json (or config.py defaults)
via GET /models to verify liveness and responsiveness before pipeline tasks launch.
"""

import sys
import httpx
from config import load_global_config, load_stage_overrides

def main():
    """Probe every configured LLM server and report which are healthy.

    Reads ``config.json``, sends each server a short request, and prints the result
    per server. The runner calls this before every stage and again before each retry.

    Returns:
        None. Exits 2 when ``config.json`` is invalid and 1 when any server is
        unreachable, so a caller can tell the two apart."""
    try:
        servers, model = load_global_config()
        overrides = load_stage_overrides()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)
    for name, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, dict)):
            print(f"ERROR: config.json stage_concurrency['{name}'] must be a number or an object.")
            sys.exit(2)
    if not isinstance(servers, list) or not servers or any(not isinstance(s, str) or not s.strip() for s in servers):
        print("ERROR: config.json must contain a non-empty 'servers' list of URLs.")
        sys.exit(2)

    print("=" * 65)
    print(f" PRE-FLIGHT CHECK: Checking {len(servers)} configured vLLM servers...")
    print(f" Expected Model: {model}")
    print("=" * 65)

    all_ok = True
    for s in servers:
        target = s.rstrip('/')
        url = f"{target}/models"
        try:
            r = httpx.get(url, timeout=7.0)
            if r.status_code == 200:
                print(f" [PASS] {target} is LIVE (HTTP 200)")
            else:
                print(f" [FAIL] {target} returned HTTP {r.status_code}")
                all_ok = False
        except Exception as e:
            print(f" [FAIL] {target} is UNREACHABLE ({type(e).__name__}: {e})")
            all_ok = False

    print("=" * 65)
    if not all_ok:
        print("ERROR: One or more vLLM servers are down or unreachable!")
        print("Please check your sbatch jobs / network or update config.json.")
        sys.exit(1)
    else:
        print(f"ALL {len(servers)} SERVERS ARE LIVE AND HEALTHY! Proceeding...")
        print("=" * 65)

if __name__ == "__main__":
    main()
