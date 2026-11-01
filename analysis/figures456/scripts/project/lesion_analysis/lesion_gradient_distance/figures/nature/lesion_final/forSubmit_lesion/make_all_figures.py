from __future__ import annotations
import os as _portable_os
from pathlib import Path as _PortablePath

def _portable_path(kind, relative):
    project = _PortablePath(_portable_os.environ['PAPER_PROJECT_ROOT'])
    root = project if kind == 'project' else project.parent
    return str(root / relative)
import subprocess
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / 'scripts'
PLOTS = ['make_lesion_longitudinal_volume.py', 'make_lesion_matched_clinical.py', 'make_lesion_matched_clinical_vertical.py', 'make_g2_spintest_vertical.py', 'make_g2_network_radar.py', 'make_g1_g2_surface_maps.py']

def main() -> None:
    for script in PLOTS:
        print(f'Running {script}')
        subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=str(SCRIPTS), check=True)
    print(f"Submission figures written to: {HERE / 'figures'}")
if __name__ == '__main__':
    main()
