import mlflow
import os
import sys
import torch
from mace.calculators import LAMMPS_MACE
from e3nn.util import jit

assert len(sys.argv) == 2, "Usage: python build_model.py <model uri>"
model_uri = f"{sys.argv[1]}/data"
with mlflow.start_run() as run:
    print(f"Starting {run.info.run_id}")
    # Download the model to a local directory
    download_dir = mlflow.artifacts.download_artifacts(model_uri, dst_path=run.info.run_id)

    # Detrmine the name of the actual model file
    downloaded = os.listdir(download_dir)
    assert len(downloaded) == 1
    model_file = os.path.join(download_dir, downloaded[0])

    model = torch.load(model_file)
    model = model.double().to("cpu")

    lammps_model = LAMMPS_MACE(model)
    lammps_model_compiled = jit.compile(lammps_model)

    lammps_model_path = os.path.join(run.info.run_id, "mace-model-lammps.pt")
    lammps_model_compiled.save(lammps_model_path)
