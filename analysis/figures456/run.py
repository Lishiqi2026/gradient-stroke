"""Portable entry point; numerical inputs and scripts are resolved relative to this folder."""
from pathlib import Path
import argparse,os,sys,subprocess,shutil,json,time,stat
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['full','prepare','core','plots'],default='full');parser.add_argument('--figure',type=int,choices=[4,5,6],default=None);args=parser.parse_args()
figure=args.figure if args.figure is not None else json.loads((ROOT/'package_config.json').read_text(encoding='utf-8'))['figure']
os.environ['PAPER_FIGURE']=str(figure)
work=ROOT/'work';work.mkdir(exist_ok=True)
os.environ['PAPER_PROJECT_ROOT']=str(work/'Gradient');os.environ['NEUROMAPS_DATA']=str(ROOT/'input_data/neuromaps-data');os.environ['MPLBACKEND']='Agg';os.environ['JOBLIB_TEMP_FOLDER']=str(work/'.joblib_temp')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
def overlay(source,dest):
    def writable_copy(src,dst):
        target=Path(dst)
        if target.exists():target.chmod(target.stat().st_mode|stat.S_IWUSR)
        shutil.copy2(src,dst)
        target.chmod(target.stat().st_mode|stat.S_IWUSR)
        return str(dst)
    if source.exists():shutil.copytree(source,dest,dirs_exist_ok=True,copy_function=writable_copy,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
def run(name,pdf=False):
    executable=os.environ.get('PDF_PYTHON',sys.executable) if pdf else sys.executable
    print('RUN',name,flush=True);start=time.time();subprocess.run([executable,str(ROOT/'scripts'/name)],check=True,env=os.environ.copy(),cwd=ROOT)
    print('FINISHED',name,round(time.time()-start,1),'seconds',flush=True)
overlay(ROOT/'input_data/work',work);overlay(ROOT/'scripts/project',work/'Gradient')
if args.stage=='plots':
    overlay(ROOT/'results/replay_data',work)
    overlay(ROOT/'results/cross_cohort_recomputed',ROOT/'cross_cohort_recomputed')
else:
    run('reproduce_fc.py');run('compute_lesion_inputs.py')
    if args.stage in ['prepare','core']:
        run('derive_core_features.py')
        print('CORE FEATURES READY IN work/; run the full pipeline to refresh inferential statistics.',flush=True)
        sys.exit(0)
    if figure in [4,5]:run('analyze_updated_fig4.py')
    else:run('derive_core_features.py')
    if figure in [5,6]:
        run('recompute_external_base.py');run('recompute_external_clinical.py')
    if figure==5:run('recompute_cross_cohort.py')
    if figure==6:
        jobs=[]
        for name in ['recompute_main_prediction.py','recompute_external_prediction.py']:
            log=(ROOT/(name.removesuffix('.py')+'.log')).open('w',encoding='utf-8')
            process=subprocess.Popen([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            jobs.append((name,process,log));print('RUN IN PARALLEL',name,flush=True)
        for name,process,log in jobs:
            code=process.wait();log.close()
            if code:raise RuntimeError(f'{name} failed with code {code}; see its log')
if figure==4:run('render_fig4.py')
if figure==5:
    run('render_external_base.py');run('render_fig5_cross.py');run('render_network_counts.py')
if figure==6:run('render_fig6.py')
run('assemble_delivery.py',pdf=True);run('write_package_results.py')
print('PACKAGE REPRODUCTION COMPLETE',figure,flush=True)
