"""Workaround for broken macOS DNS resolver.

`socket.getaddrinfo` fails with errno 8 even though `host` and `nslookup`
return valid IPs. We monkey-patch `socket.getaddrinfo` to call out to `host`
as a fallback if the normal resolution fails. Call `install()` once at process
start.
"""
from __future__ import annotations
import socket
import subprocess
import threading

_lock = threading.Lock()
_cache: dict[str, list[str]] = {}
_orig_getaddrinfo = socket.getaddrinfo


def _host_resolve(host: str) -> list[str]:
    with _lock:
        if host in _cache:
            return _cache[host]
        try:
            out = subprocess.check_output(["host", host], text=True, timeout=8)
        except Exception:
            return []
        ips = []
        for line in out.splitlines():
            if "has address" in line:
                ip = line.rsplit(" ", 1)[-1].strip()
                ips.append(ip)
        if ips:
            _cache[host] = ips
        return ips


def _patched_getaddrinfo(host, port, *args, **kwargs):
    try:
        return _orig_getaddrinfo(host, port, *args, **kwargs)
    except socket.gaierror as e:
        # Fall back to host(1) lookup
        ips = _host_resolve(host)
        if not ips:
            raise
        # Build manual address-info tuples
        out = []
        for ip in ips:
            try:
                out.extend(_orig_getaddrinfo(ip, port, *args, **kwargs))
            except Exception:
                continue
        if out:
            return out
        raise e


def install():
    if socket.getaddrinfo is _patched_getaddrinfo:
        return
    socket.getaddrinfo = _patched_getaddrinfo
