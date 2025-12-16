# Step 7C – COMPASS execution (WSL environment)

## Overview

This step performs **metabolic flux analysis** using COMPASS (YosefLab) on
micropooled myogenic populations generated in Step 7A.

Due to dependency constraints (libSBML and Gurobi), COMPASS was executed
in a **Linux (WSL) environment**, while upstream preprocessing was performed
on native Windows.

---

## Rationale for using WSL

COMPASS relies on:

- `python-libsbml`
- Gurobi Optimizer

These dependencies are **not reliably supported on native Windows Python**.
To ensure solver stability and reproducibility, all COMPASS analyses were
performed in **WSL (Ubuntu)**.

---

## Environment setup

### Enter WSL

```bash
wsl


All subsequent commands were executed inside WSL.

Create COMPASS environment
conda create -n py_compass python=3.8 -y
conda activate py_compass


Python 3.8 was used to ensure compatibility with COMPASS and libSBML.

Install dependencies
pip install numpy pandas scipy scikit-learn tqdm
pip install python-libsbml
pip install git+https://github.com/yoseflab/Compass.git


Installation was validated by:

which compass

python - << 'EOF'
import libsbml
import compass
print("libsbml OK:", libsbml.__version__)
print("compass OK")
EOF


Expected output:

libsbml OK
compass OK

Gurobi configuration

Gurobi was installed at the system level with an academic license.

License check:

gurobi_cl --license


Expected output:

Academic license - for non-commercial use only


Note: Gurobi does not need to be installed inside the conda environment.
COMPASS accesses Gurobi via gurobipy, which links to the system installation.

Running COMPASS

The COMPASS run directory was prepared in Step 7B.

From within WSL:

cd /mnt/c/Users/User/Desktop/Single\ cell\ for\ CoQ/Data_raw/step7_compass/compass_run_myogenic
conda activate py_compass
bash run_compass.sh


Outputs are written to:

compass_out/

Analysis scope

COMPASS was applied to micropooled myogenic populations, including:

MuSC

Type I myofibers

Type II myofibers

Specialized myofibers

Downstream analyses focus on the ubiquinone (CoQ10) biosynthesis pathway
and compare metabolic potential across age and intervention groups.

Reproducibility note

Preprocessing and micropooling were performed on Windows.

Metabolic flux analysis was executed in WSL.

All environment setup and execution steps are documented to ensure
full reproducibility.
