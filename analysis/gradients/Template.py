from pathlib import Path
import contextlib
import csv
import io
import json
import numpy as np
import brainspace
from brainspace.gradient.alignment import procrustes_alignment

ROOT = Path(r'F:\lishiqi\预处理数据\cortex\Vertex (18715, 300)')
OUTPUT = Path(r'F:\lishiqi\预处理数据\cortex\Template.npy')
REPORT_DIR = Path(__file__).resolve().parent
N_ITER = 10
TOL = 1e-5
EXPECTED_SHAPE = (18715, 300)

def main():
    if OUTPUT.exists():
        raise FileExistsError(f'模板已存在，未覆盖：{OUTPUT}')
    files = []
    for group in ('Control', 'TMS'):
        for period in ('before', 'after'):
            folder = ROOT / group / period
            if not folder.is_dir():
                raise FileNotFoundError(f'目录不存在：{folder}')
            found = sorted(folder.glob('*.npy'))
            if not found:
                raise ValueError(f'目录没有 npy 文件：{folder}')
            print(f'{group}/{period}: {len(found)}')
            files.extend((group, period, path) for path in found)

    arrays = []
    for group, period, path in files:
        try:
            x = np.load(path, mmap_mode='r', allow_pickle=False)
            if x.shape != EXPECTED_SHAPE:
                raise ValueError(f'维度为 {x.shape}，预期 {EXPECTED_SHAPE}')
            # 复制为独立数组，仅取前五个梯度。
            first_five = np.array(x[:, :5], dtype=np.float64, copy=True)
            if not np.isfinite(first_five).all():
                raise ValueError('前五个梯度含 NaN 或无穷值')
            arrays.append(first_five)
        except Exception as exc:
            raise ValueError(f'输入检查失败：{path}：{exc}') from exc

    print(f'有效扫描：{len(arrays)}；开始 BrainSpace 广义 Procrustes 对齐')
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        aligned, template = procrustes_alignment(
            arrays, n_iter=N_ITER, tol=TOL, return_reference=True, verbose=True
        )
    iterations = capture.getvalue()
    print(iterations, end='')
    if template.shape != (18715, 5) or not np.isfinite(template).all():
        raise ValueError('生成的模板维度不正确或包含非有限值')

    # 使用独占创建，避免覆盖已有模板。
    with OUTPUT.open('xb') as output_file:
        np.save(output_file, template)
    check = np.load(OUTPUT, mmap_mode='r', allow_pickle=False)
    if check.shape != (18715, 5):
        raise ValueError('保存后的模板维度不正确')
    with (REPORT_DIR / 'participants.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'timepoint', 'filename', 'path'])
        writer.writerows((g, t, p.name, str(p)) for g, t, p in files)
    report = {'brainspace_version': brainspace.__version__, 'n_scans': len(files),
              'shape': list(template.shape), 'n_iter_max': N_ITER, 'tol': TOL,
              'weighting': 'equal per scan', 'initial_reference': 'first file in sorted order',
              'iterations': iterations.splitlines(), 'output': str(OUTPUT)}
    (REPORT_DIR / 'template_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'模板已保存：{OUTPUT}，维度：{template.shape}')
    print(f'名单和报告：{REPORT_DIR}')

if __name__ == '__main__':
    main()
