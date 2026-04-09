import os


def configure_runtime_env():
    """
    Reduce OpenMP runtime conflicts on Windows by:
    1) prioritizing active conda env DLL locations
    2) removing Intel oneAPI compiler bin from PATH for this process
    """
    path_sep = os.pathsep
    path_entries = os.environ.get("PATH", "").split(path_sep)

    filtered = []
    for entry in path_entries:
        low = entry.lower()
        if "intel\\oneapi\\compiler" in low:
            continue
        filtered.append(entry)

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    preferred = []
    if conda_prefix:
        preferred = [
            conda_prefix,
            os.path.join(conda_prefix, "Library", "bin"),
            os.path.join(conda_prefix, "Scripts"),
        ]

    seen = set()
    merged = []
    for p in preferred + filtered:
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        merged.append(p)

    os.environ["PATH"] = path_sep.join(merged)
    os.environ.setdefault("CONDA_DLL_SEARCH_MODIFICATION_ENABLE", "1")
    # Fallback for environments that still load duplicated Intel OpenMP runtimes.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

