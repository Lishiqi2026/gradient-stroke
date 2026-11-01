from __future__ import annotations
'Re-render panels b/c/d/f from saved source tables after text-only revisions.'
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import importlib.util
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / 'recompute_bcdf_with_provided_reference.py'

def load_module():
    specification = importlib.util.spec_from_file_location('bcdf_recomputed', MODULE_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f'Could not load {MODULE_PATH}')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module

def main() -> None:
    module = load_module()
    module.make_figures(pd.read_csv(HERE / 'recomputed_subject_metrics.csv', dtype={'subject_id': str}), pd.read_csv(HERE / 'recomputed_group_models.csv'), pd.read_csv(HERE / 'panel_c_group_distance_bin_summary.csv'), pd.read_csv(HERE / 'recomputed_matched_tests.csv'), pd.read_csv(HERE / 'recomputed_matched_pair_values.csv'))
if __name__ == '__main__':
    main()
