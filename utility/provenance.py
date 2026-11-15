"""Experiment and checkpoint provenance (F14). Read-only identity records; no behaviour change.

Checkpoint compatibility policy
-------------------------------
* A checkpoint saved with provenance is loaded into an environment only if its controller
  INTERFACE matches: agent_obs feature names/order, node feature names/order, feeder size,
  DESS bus mapping, and the EFFECTIVE observation-normalization scales. Any difference raises.
* Configuration hash, network/topology files and data hashes are RECORDED but not compared:
  zero-shot evaluation on another topology or day split is a legitimate use of a checkpoint.
* A checkpoint without provenance was saved before F14, hence before the F3 normalization
  (PV and curtailment were then un-normalized). Loading it into an environment that uses the
  F3 normalization (schema 2) is REFUSED: its inputs would be mis-scaled. (Before F3 such
  checkpoints were accepted with status 'legacy_no_provenance' and a warning.) Loading without
  an environment is not checked ('not_checked').
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 1
NORMALIZER_FIELDS = ("max_time_index", "max_price", "max_load_kw", "max_pv_kw",
                     "max_grid_import_kw", "max_voltage_deviation", "max_line_current_pu",
                     "max_curtailment_kw")
NORMALIZATION_SCHEMA = 2  # F3: explicit PV (per bus) and feeder curtailment references
SOURCE_DIRS = ("configs", "environments", "models", "training", "utility")
REPO = Path(__file__).resolve().parents[1]
_FILE_CACHE = {}


def _json_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_record(path):
    """Path, size, mtime and sha256 (raw and LF-normalized) of one input file; cached by mtime."""
    p = Path(path)
    if not p.is_file():
        return dict(path=str(p.resolve()), exists=False)
    stat = p.stat()
    key = (str(p.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _FILE_CACHE:
        raw = p.read_bytes()
        _FILE_CACHE[key] = dict(sha256=hashlib.sha256(raw).hexdigest(),
                                sha256_lf=hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest())
    return dict(path=key[0], exists=True, size_bytes=stat.st_size, modified_ns=stat.st_mtime_ns, **_FILE_CACHE[key])


def input_files(env):
    """Every file referenced in config['paths'] (network CSVs incl. the PF line file, data)."""
    return {k: file_record(v) for k, v in env.config.get("paths", {}).items()
            if isinstance(v, (str, Path)) and Path(v).suffix}


def normalization_identity(env):
    """Effective scales actually used by the agent-observation and graph-node normalizers."""
    def scales(normalizer):
        return {f: float(getattr(normalizer, f)) for f in NORMALIZER_FIELDS}
    ident = dict(schema=NORMALIZATION_SCHEMA,
                 normalize_observations=bool(getattr(env, "normalize_observations", True)),
                 agent_obs=scales(env.normalizer), graph_nodes=scales(env.graph_builder.normalizer))
    ident["sha256"] = _json_sha({k: v for k, v in ident.items() if k != "sha256"})
    return ident


def interface_identity(env):
    graph = env.config.get("graph", {})
    return dict(agent_obs_dim=int(env.agent_obs_dim),
                agent_obs_features=list(getattr(env, "agent_obs_features", [])),
                node_feature_dim=int(env.observation_space["x"].shape[-1]),
                node_features=list(graph.get("node_features", [])),
                num_buses=int(env.num_buses), dess_buses=[int(b) for b in env.dess_buses])


def source_files():
    out = {}
    for d in SOURCE_DIRS:
        for p in sorted((REPO / d).rglob("*.py")):
            if "__pycache__" not in p.parts:
                out[p.relative_to(REPO).as_posix()] = hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return dict(files=out, sha256=_json_sha(out))


def configuration_sha256(env):
    from utility.run_manifest import serial
    return _json_sha(serial(env.config))


def checkpoint_provenance(env):
    """Provenance block stored inside checkpoints."""
    return dict(schema=SCHEMA, interface=interface_identity(env), normalization=normalization_identity(env),
                configuration_sha256=configuration_sha256(env),
                topology_case=(env.config.get("topology_case") or {}).get("case_name", "TP1"),
                input_files={k: v.get("sha256") for k, v in input_files(env).items()},
                source_files_sha256=source_files()["sha256"])


def check_checkpoint_provenance(stored, env, path=""):
    """Return 'verified'; raise ValueError on an interface/normalization mismatch."""
    if stored is None:
        # F3: provenance-less checkpoints predate F14 and therefore the F3 normalization.
        raise ValueError(f"Checkpoint {path} has no provenance: it was saved before F14 and therefore "
                         f"before the F3 observation normalization (schema {NORMALIZATION_SCHEMA}); its PV and "
                         "curtailment inputs would be mis-scaled. Retrain with the current code.")
    current = dict(interface=interface_identity(env), normalization=normalization_identity(env))
    problems = []
    for key, value in current["interface"].items():
        if stored.get("interface", {}).get(key) != value:
            problems.append(f"interface.{key}: checkpoint={stored.get('interface', {}).get(key)!r} env={value!r}")
    for part in ("agent_obs", "graph_nodes"):
        for f, v in current["normalization"][part].items():
            s = stored.get("normalization", {}).get(part, {}).get(f)
            if s is None or not np.isclose(float(s), v, rtol=0, atol=0):
                problems.append(f"normalization.{part}.{f}: checkpoint={s!r} env={v!r}")
    if stored.get("normalization", {}).get("normalize_observations") != current["normalization"]["normalize_observations"]:
        problems.append("normalization.normalize_observations differs")
    if stored.get("normalization", {}).get("schema") != NORMALIZATION_SCHEMA:
        problems.append(f"normalization.schema: checkpoint={stored.get('normalization', {}).get('schema')!r} "
                        f"env={NORMALIZATION_SCHEMA} (saved before the F3 normalization)")
    if problems:
        raise ValueError(f"Checkpoint {path} is incompatible with this environment:\n  " + "\n  ".join(problems))
    return "verified"
