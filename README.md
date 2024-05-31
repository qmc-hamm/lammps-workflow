# lammps-workflow
Automated Workflow for LAMMPS Simulations

## Steps
1. Download a mace model from the repository and convert it to a LAMMPS .pt model


## 1. Create the LAMMPS .pt model
This operation requires a GPU so we will run it with the mlflow-slurm plugin
against the `gpu-slurm.json` configuration file.

```bash
mlflow run --backend slurm \
    --backend-config gpu-slurm.json \
    --entry-point build-model .
```
