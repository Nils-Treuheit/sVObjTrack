"""
Shared model registry for sVObjTrack.

Resolves model paths, downloads from HuggingFace Hub when needed,
and discovers external venv paths relative to VLMEXPERIMENTS_ROOT.
"""
import os
from pathlib import Path
from functools import lru_cache

# ---------------------------------------------------------------------------
# Root paths — all external models live under this directory.
# Override via environment variable if your layout differs.
# ---------------------------------------------------------------------------
VLMEXPERIMENTS_ROOT = Path(os.environ.get(
    'VLMEXPERIMENTS_ROOT',
    '/mnt/HDD1/Project_Code/VLMexperiments/VLMcollection'))

WORKSPACE_ROOT = Path(os.environ.get('WORKSPACE_ROOT', '')).resolve() if os.environ.get('WORKSPACE_ROOT') else None

# Auto-detect workspace root by walking up from __file__ looking for a models/ dir
if WORKSPACE_ROOT is None:
    _p = Path(__file__).resolve().parent
    for _ in range(10):
        if (_p / 'models').is_dir() or (_p / 'src').is_dir():
            WORKSPACE_ROOT = _p
            break
        _p = _p.parent
    if WORKSPACE_ROOT is None:
        WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

MODELS_DIR = WORKSPACE_ROOT / 'models'

# ---------------------------------------------------------------------------
# External venvs — auto-discovered relative to VLMEXPERIMENTS_ROOT
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _find_venv(base: Path, venv_name: str = '.venv') -> Path:
    """Find a venv directory, returning Path or raising."""
    candidate = base / venv_name
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f'Venv not found: {candidate}')


def rfdetr_venv() -> Path:
    return _find_venv(VLMEXPERIMENTS_ROOT / 'rfdetr')


def metadepth_venv() -> Path:
    return _find_venv(VLMEXPERIMENTS_ROOT / 'metadepth')


def sam3_venv() -> Path:
    # SAM3 dir may be named 'sam3' or 'sam3_1'
    for name in ('sam3_1', 'sam3'):
        candidate = VLMEXPERIMENTS_ROOT / name
        if candidate.is_dir():
            return _find_venv(candidate)
    raise FileNotFoundError('SAM3 venv not found')


def la_project() -> Path:
    return VLMEXPERIMENTS_ROOT / 'locate_anything'


def la_trt_dir() -> Path:
    return la_project() / 'model' / 'tensorRT'


# ---------------------------------------------------------------------------
# HuggingFace Hub downloads
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _hf_download(repo_id: str, filename: str, subfolder: str = '') -> Path:
    """Download a file from HuggingFace Hub, caching the result."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder or None,
        local_dir=str(VLMEXPERIMENTS_ROOT / '_hf_cache' / repo_id.replace('/', '_')),
    )
    return Path(path)


# ---------------------------------------------------------------------------
# Model checkpoint resolution
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def resolve_checkpoint(model_name: str) -> Path:
    """Resolve a model checkpoint path.

    Priority:
      1. Exact file path (if model_name contains '/' or ends with .pt/.pth/.pth)
      2. File in MODELS_DIR
      3. HuggingFace Hub download
      4. FileNotFoundError
    """
    # If it's already an absolute path or contains a directory separator, use as-is
    p = Path(model_name)
    if p.is_absolute() and p.exists():
        return p

    # Check MODELS_DIR
    local = MODELS_DIR / model_name
    if local.exists():
        return local

    # Try HuggingFace Hub based on file extension / known patterns
    hf_map = {
        'yolov8s-worldv2.pt': ('ultralytics/yolov8s-worldv2', 'yolov8s-worldv2.pt'),
        'yolov8s-world.pt': ('ultralytics/yolov8s-world', 'yolov8s-world.pt'),
    }
    if model_name in hf_map:
        repo, fname = hf_map[model_name]
        return _hf_download(repo, fname)

    raise FileNotFoundError(
        f'Model "{model_name}" not found locally at {local} '
        f'and no HuggingFace Hub mapping exists.')


# ---------------------------------------------------------------------------
# SAM 3.1 checkpoint — always from HuggingFace
# ---------------------------------------------------------------------------

def sam3_checkpoint() -> Path:
    """Get or download the SAM 3.1 multiplex checkpoint."""
    # Check local cache first
    local_cache = VLMEXPERIMENTS_ROOT / '_hf_cache' / 'facebook_sam3.1' / 'sam3.1_multiplex.pt'
    if local_cache.exists():
        return local_cache
    return _hf_download('facebook/sam3.1', 'sam3.1_multiplex.pt')


# ---------------------------------------------------------------------------
# MetaDepth / HYDEN-MoGeV2 checkpoint — always from HuggingFace
# ---------------------------------------------------------------------------

def metadepth_checkpoint() -> Path:
    """Get or download the HYDEN-MoGeV2 metric point checkpoint."""
    local_cache = (VLMEXPERIMENTS_ROOT / '_hf_cache' / 'facebook_hyden-mogev2-metric-point'
                   / 'hyden_mogev2_metric_point_vitl_fp32_f1066593896.pth')
    if local_cache.exists():
        return local_cache
    return _hf_download(
        'facebook/hyden-mogev2-metric-point',
        'hyden_mogev2_metric_point_vitl_fp32_f1066593896.pth')


# ---------------------------------------------------------------------------
# YOLO base weights — resolve from MODELS_DIR or download
# ---------------------------------------------------------------------------

def yolo_base_weights(name: str) -> Path:
    """Resolve YOLO base weights from MODELS_DIR."""
    return resolve_checkpoint(name)
