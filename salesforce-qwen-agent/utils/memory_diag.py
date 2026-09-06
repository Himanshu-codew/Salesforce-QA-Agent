"""
Memory diagnostics — safe RSS logging at key lifecycle points.

Logs RSS (Resident Set Size) in MB at startup, after RAG init,
before/after Qwen, before/after Salesforce, and at request completion.

NEVER logs: OAuth tokens, client secrets, passwords, security tokens,
or any sensitive credentials.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)


def _get_rss_mb() -> float:
    """Return current RSS in MB. Works on Linux/macOS/Windows."""
    # Windows: use psutil or ctypes
    if os.name == "nt":
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            pass
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_uint32),
                    ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, os.getpid())
            if handle:
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    kernel32.CloseHandle(handle)
                    return counters.WorkingSetSize / (1024 * 1024)
                kernel32.CloseHandle(handle)
        except Exception:
            pass
        return 0.0
    else:
        # Linux/macOS: try /proc/self/status first
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            pass
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except (ImportError, Exception):
            pass
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024.0  # Linux: KB -> MB
        except Exception:
            pass
    return 0.0


def log_memory(tag: str) -> float:
    """Log memory RSS at a lifecycle point. Returns RSS in MB."""
    rss = _get_rss_mb()
    if rss > 0:
        logger.info(f"[MEM] {tag}: RSS={rss:.1f}MB")
    else:
        logger.info(f"[MEM] {tag}: RSS=unavailable")
    return rss


def log_startup() -> None:
    """Log memory at application startup."""
    log_memory("startup")


def log_after_rag_init() -> None:
    """Log memory after RAG embedding model + ChromaDB init."""
    log_memory("after RAG init")


def log_before_qwen() -> None:
    """Log memory before a Qwen LLM request."""
    log_memory("before Qwen")


def log_after_qwen() -> None:
    """Log memory after a Qwen LLM response."""
    log_memory("after Qwen")


def log_before_salesforce() -> None:
    """Log memory before a Salesforce/MCP call."""
    log_memory("before Salesforce")


def log_after_salesforce() -> None:
    """Log memory after a Salesforce/MCP call."""
    log_memory("after Salesforce")


def log_request_complete(session_id: str = "") -> None:
    """Log memory at request completion."""
    rss = log_memory(f"request complete (session={session_id})")
    # Force garbage collection to reclaim temporary objects
    import gc
    gc.collect()
    rss_after = _get_rss_mb()
    if rss_after > 0 and rss > 0 and rss_after < rss:
        logger.info(f"[MEM] after gc.collect(): RSS={rss_after:.1f}MB (freed {rss - rss_after:.1f}MB)")
