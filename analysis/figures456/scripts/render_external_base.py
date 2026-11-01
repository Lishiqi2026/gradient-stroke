from pathlib import Path
import os,sys,subprocess
os.environ['MPLBACKEND']='Agg'
RUN=Path(__file__).resolve().parents[1];HERE=RUN/'work/Gradient/lesion_analysis/external_validation/final'
scripts=['regenerate_gradient_panels_forai.py','regenerate_seven_network_organization_forai.py','regenerate_component_variance_forai.py','regenerate_clinical_score_forai.py','regenerate_clinical_bridge_forai.py','regenerate_clinical_association_forai.py','regenerate_clinical_association_network_glass_brains_forai.py']
if '--resume-clinical' in sys.argv:scripts=scripts[4:]
if '--resume-score' in sys.argv:scripts=scripts[3:]
for name in scripts:
    print('RENDER',name,flush=True)
    extra=['--palette','coordinated'] if name=='regenerate_component_variance_forai.py' else []
    subprocess.run([sys.executable,str(HERE/name),*extra],check=True,env=os.environ.copy())
print('EXTERNAL BASE PANELS COMPLETE',flush=True)
