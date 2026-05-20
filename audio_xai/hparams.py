"""YAML hyperparameter loading for train_classifier.py, predict.py, and attack.py.

Usage in a script
-----------------
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config:
        from audio_xai.hparams import train_defaults
        parser.set_defaults(**train_defaults(pre_args.config))
    args = parser.parse_args()

CLI flags always override values from the YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for --config: pip install pyyaml") from exc
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _nonnull(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def train_defaults(cfg_path: str | Path) -> dict[str, Any]:
    """Return argparse dest-name defaults for train_classifier.py from a YAML config."""
    c       = _load_yaml(cfg_path)
    data    = c.get("data", {})
    model   = c.get("model", {})
    tr      = c.get("training", {})
    cluster = c.get("cluster", {})

    raw: dict[str, Any] = {
        "seed":            c.get("seed"),
        "clip_seconds":    c.get("clip_seconds"),
        # data
        "data_root":       data.get("root"),
        "train_csv":       data.get("train_csv"),
        "val_csv":         data.get("val_csv"),
        "real_subdir":     data.get("real_subdir"),
        "fake_subdir":     data.get("fake_subdir"),
        "val_frac":        data.get("val_frac"),
        "max_per_class":   data.get("max_per_class"),
        # model
        "model":           model.get("name"),
        "vggish_ckpt":     model.get("vggish_ckpt"),
        # training
        "batch_size":      tr.get("batch_size"),
        "epochs":          tr.get("epochs"),
        "lr":              tr.get("lr"),
        "num_workers":     tr.get("num_workers"),
        "out_dir":         tr.get("out_dir"),
        # cluster
        "num_nodes":       cluster.get("num_nodes"),
        "devices":         cluster.get("devices"),
        "strategy":        cluster.get("strategy"),
        "no_auto_requeue": cluster.get("no_auto_requeue"),
    }
    out = _nonnull(raw)
    for key in ("data_root", "train_csv", "val_csv", "out_dir"):
        if key in out:
            out[key] = Path(out[key])
    return out


def predict_defaults(cfg_path: str | Path) -> dict[str, Any]:
    """Return argparse dest-name defaults for predict.py from a YAML config."""
    c     = _load_yaml(cfg_path)
    data  = c.get("data", {})
    model = c.get("model", {})
    pred  = c.get("predict", {})

    raw: dict[str, Any] = {
        "seed":        c.get("seed"),
        "clip_seconds": c.get("clip_seconds"),
        # data
        "data_root":   data.get("root"),
        # model
        "model_type":  model.get("name"),
        "model_id":    model.get("model_id"),
        "checkpoint":  model.get("checkpoint"),
        # predict
        "split":       pred.get("split"),
        "n_samples":   pred.get("n_samples"),
        "batch_size":  pred.get("batch_size"),
        "sample_rate": pred.get("sample_rate"),
        "log_dir":     pred.get("log_dir"),
        "device":      pred.get("device"),
    }
    out = _nonnull(raw)
    for key in ("data_root", "checkpoint", "log_dir"):
        if key in out:
            out[key] = Path(out[key])
    return out


def pgd_attack_defaults(cfg_path: str | Path) -> dict[str, Any]:
    """Return argparse dest-name defaults for pgd_attack.py from a YAML config.

    Reads PGD-specific params from the [pgd_attack] section if present,
    falling back to [attack] for shared keys (n_attack_samples, micro_batch,
    lambda_aud, lambda_pred, device).  Model and data sections are shared with
    the existing YAML configs (predict_ast.yaml etc.) so no new files needed.
    """
    c      = _load_yaml(cfg_path)
    data   = c.get("data", {})
    model  = c.get("model", {})
    pred   = c.get("predict", {})
    pgd    = c.get("pgd_attack", {})
    atk    = c.get("attack", {})   # fallback for shared keys

    raw: dict[str, Any] = {
        "seed":               c.get("seed"),
        "clip_seconds":       c.get("clip_seconds"),
        # data
        "data_root":          data.get("root"),
        # model
        "model_type":         model.get("name"),
        "model_id":           model.get("model_id"),
        "checkpoint":         model.get("checkpoint"),
        # split
        "split":              pred.get("split"),
        "sample_rate":        pred.get("sample_rate"),
        # attack — pgd_attack section takes priority, falls back to attack section
        "n_attack_samples":   pgd.get("n_attack_samples")   or atk.get("n_attack_samples"),
        "attack_micro_batch": pgd.get("micro_batch")        or atk.get("micro_batch"),
        "eps":                pgd.get("eps"),
        "alpha":              pgd.get("alpha"),
        "num_iter":           pgd.get("num_iter"),
        "lambda_aud":         pgd.get("lambda_aud")         or atk.get("lambda_aud"),
        "lambda_pred":        pgd.get("lambda_pred")        or atk.get("lambda_pred"),
        "log_dir":            pgd.get("log_dir")            or atk.get("log_dir") or pred.get("log_dir"),
        "device":             pgd.get("device")             or atk.get("device")  or pred.get("device"),
    }
    out = _nonnull(raw)
    for key in ("data_root", "checkpoint", "log_dir"):
        if key in out:
            out[key] = Path(out[key])
    return out


def attack_defaults(cfg_path: str | Path) -> dict[str, Any]:
    """Return argparse dest-name defaults for attack.py from a YAML config."""
    c      = _load_yaml(cfg_path)
    data   = c.get("data", {})
    model  = c.get("model", {})
    pred   = c.get("predict", {})
    attack = c.get("attack", {})

    raw: dict[str, Any] = {
        "seed":               c.get("seed"),
        "clip_seconds":       c.get("clip_seconds"),
        # data
        "data_root":          data.get("root"),
        # model
        "model_type":         model.get("name"),
        "model_id":           model.get("model_id"),
        "checkpoint":         model.get("checkpoint"),
        # split (shared with predict section)
        "split":              pred.get("split"),
        "sample_rate":        pred.get("sample_rate"),
        # attack
        "n_attack_samples":   attack.get("n_attack_samples"),
        "attack_micro_batch": attack.get("micro_batch"),
        "n_steps":            attack.get("n_steps"),
        "lr":                 attack.get("lr"),
        "lambda_aud":         attack.get("lambda_aud"),
        "lambda_pred":        attack.get("lambda_pred"),
        "pred_margin":        attack.get("pred_margin"),
        "log_dir":            attack.get("log_dir") or pred.get("log_dir"),
        "device":             attack.get("device") or pred.get("device"),
    }
    out = _nonnull(raw)
    for key in ("data_root", "checkpoint", "log_dir"):
        if key in out:
            out[key] = Path(out[key])
    return out