import mlflow
import os
import sys
import torch
from mace.calculators import LAMMPS_MACE
from e3nn.util import jit

def main(model_uri: str, local_dir: str):
    with mlflow.start_run() as run:
        print(f"Starting {run.info.run_id}")
        # Download the model to a local directory named after the parent run
        download_dir = mlflow.artifacts.download_artifacts(model_uri, dst_path=local_dir)

        # Detrmine the name of the actual model file
        downloaded = os.listdir(download_dir)
        assert len(downloaded) == 1
        model_file = os.path.join(download_dir, downloaded[0])

        model = torch.load(model_file)
        model = model.double().to("cpu")

        lammps_model = LAMMPS_MACE(model)
        lammps_model_compiled = jit.compile(lammps_model)

        lammps_model_path = os.path.join(local_dir, "mace-model-lammps.pt")
        lammps_model_compiled.save(lammps_model_path)

        mlflow.log_artifact(lammps_model_path)

if __name__ == '__main__':
    assert len(sys.argv) == 3, "Usage: python build_model.py <model uri> <parent runid>"
    model_uri = f"{sys.argv[1]}/data"
    parent_runid = sys.argv[2]

    main(model_uri, local_dir=parent_runid)
