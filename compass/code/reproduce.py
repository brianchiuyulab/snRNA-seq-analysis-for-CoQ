"""Run the complete deposited-score analysis and figure pipeline."""
from pathlib import Path
import subprocess,sys
here=Path(__file__).resolve().parent
root=here.parent
required=['all21_metadata.tsv','donor_inventory.tsv','model_coverage.tsv',
          'bi13_metadata.tsv','all21_penalties.tsv.gz','bi13_penalties.tsv.gz',
          'reaction_metadata.csv.gz','all21_expression_cpm.tsv.gz',
          'frozen_musc_membership.tsv.gz','depth_control/profiles.json',
          'depth_control/pilot/reactions.tsv','depth_control/remaining/reactions.tsv']
missing=[str(root/'data'/p) for p in required if not (root/'data'/p).is_file()]
if missing:
    raise SystemExit('Missing companion inputs. Place the matching data directory beside code/.\n'+'\n'.join(missing))
for directory in ['tables','figures','manuscript','provenance']:
    (root/directory).mkdir(exist_ok=True)
for name in ['analyze.py','figures.py','global_ranking.py','qc_age_audit.py','depth_control.py']:
    subprocess.run([sys.executable,str(here/name)],check=True)
