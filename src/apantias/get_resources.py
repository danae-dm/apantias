import os

import psutil


def _parse_slurm_mem(mem_str: str) -> int | None:
    """Parse a SLURM memory value into an integer number of megabytes.
    SLURM values are often given as plain numbers (MB) or with a unit suffix.
    """
    if not mem_str:
        return None
    mem_str = mem_str.strip().upper()
    mb = 0.0
    try:
        if mem_str.endswith("G"):
            mb = float(mem_str[:-1]) * 1024.0          # GB -> MB
        elif mem_str.endswith("M"):
            mb = float(mem_str[:-1])                   # already MB
        elif mem_str.endswith("K"):
            mb = float(mem_str[:-1]) / 1024.0          # KB -> MB
        else:
            # SLURM's default unit is MB
            mb = float(mem_str)
    except ValueError:
        return None
    return int(mb)

def _get_cgroup_memory_limit() -> int | None:
    for path in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    val = f.read().strip()
                if val and val != "max":
                    limit = int(val)
                    # Some systems return a very high number (e.g., 9223372036854771712) when no limit is set
                    if limit < 9000000000000000000:
                        return int(limit / (1024.0**2))  # bytes -> MB
            except Exception:
                pass
    return None

def get_resources() -> tuple[int, int]:
    """
    Detect the resource allocations (CPUs, Memory) available to the current process,
    respecting SLURM environment variables, container cgroup limits, and system fallbacks.

    Returns:
        cpus: number of available CPUs
        memory_mb: available memory in megabytes (integer)
    """
    # 1. CPU allocation
    cpus = None
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        try:
            cpus = int(slurm_cpus)
        except ValueError:
            pass

    if cpus is None and hasattr(os, "sched_getaffinity"):
        try:
            cpus = len(os.sched_getaffinity(0))
        except Exception:
            pass

    if cpus is None or cpus <= 0:
        physical_cores = psutil.cpu_count(logical=False)
        cpus = physical_cores if physical_cores else (os.cpu_count() or 1)

    # 2. Memory allocation (in MB throughout)
    memory_mb = None

    slurm_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem:
        memory_mb = _parse_slurm_mem(slurm_mem)

    if memory_mb is None:
        slurm_mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
        if slurm_mem_per_cpu:
            per_cpu = _parse_slurm_mem(slurm_mem_per_cpu)
            if per_cpu is not None:
                memory_mb = per_cpu * cpus

    if memory_mb is None:
        memory_mb = _get_cgroup_memory_limit()

    if memory_mb is None:
        memory_mb = int(psutil.virtual_memory().total / (1024.0**2))  # bytes -> MB

    return cpus, memory_mb
