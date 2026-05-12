import os
import sys

import requests

# Reward gateway base URL, e.g. the proxy you ran from searchagent_scripts/proxy.
# Override with: REWARD_GATEWAY=http://your-host:8000 python test_proxy.py
BASE_URL = os.environ.get("REWARD_GATEWAY", "http://127.0.0.1:8000").rstrip("/")
N_REWARD_BACKENDS = int(os.environ.get("REWARD_BACKENDS", "8"))

URLS = [f"{BASE_URL}/reward{i}/v1/models" for i in range(1, N_REWARD_BACKENDS + 1)]

TIMEOUT = 10


def test_urls(urls):
    failed = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.ok:
                print(f"[OK]   {url}  (status={resp.status_code})")
            else:
                print(f"[FAIL] {url}  (status={resp.status_code})")
                failed.append((url, f"HTTP {resp.status_code}"))
        except requests.ConnectionError:
            print(f"[FAIL] {url}  (connection refused)")
            failed.append((url, "connection refused"))
        except requests.Timeout:
            print(f"[FAIL] {url}  (timeout after {TIMEOUT}s)")
            failed.append((url, f"timeout after {TIMEOUT}s"))
        except requests.RequestException as e:
            print(f"[FAIL] {url}  ({e})")
            failed.append((url, str(e)))

    print("\n" + "=" * 60)
    if failed:
        print(f"\n{len(failed)} URL(s) not reachable:\n")
        for url, reason in failed:
            print(f"  - {url}  ({reason})")
        sys.exit(1)
    else:
        print(f"\nAll {len(urls)} URLs are reachable.")


if __name__ == "__main__":
    test_urls(URLS)
