"""Run the complete deposited-score analysis and figure pipeline."""
from pathlib import Path
import subprocess,sys
here=Path(__file__).resolve().parent
for name in ['analyze.py','figures.py','global_ranking.py','qc_age_audit.py','depth_control.py']:
    subprocess.run([sys.executable,str(here/name)],check=True)
