from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import matplotlib.pyplot as plt
import pandas as pd
from recompute_bcdf_with_provided_reference import OUT_DIR, plot_panel_c, save_bundle

def main() -> None:
    summary = pd.read_csv(OUT_DIR / 'panel_c_group_distance_bin_summary.csv')
    (fig, axis) = plt.subplots(figsize=(3.65, 3.05))
    plot_panel_c(axis, summary, panel_label=False)
    fig.tight_layout(pad=0.7)
    save_bundle(fig, 'panel_c_recomputed_provided_reference')
if __name__ == '__main__':
    main()
