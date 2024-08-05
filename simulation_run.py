import mlflow
import os

tracking_client = mlflow.tracking.MlflowClient()

def build_model(parent_run_id: str, experiment_id: int, backend_config: str, model_uri: str):
    p = mlflow.projects.run(
        uri=os.path.dirname(os.path.realpath(__file__)),
        entry_point="build-model",
        run_name="Build Model from MACE Model"
        parameters={
            "model_uri": model_uri,
            "parent_run_id": parent_run_id
        },
        experiment_id=experiment_id,
        synchronous=True,
        backend="slurm",
        backend_config=backend_config
    )

def run(provide_run_id=None):
    provided_run_id = os.environ.get("MLFLOW_RUN_ID", None)
    with mlflow.start_run(run_id=provided_run_id) as run:
        if not provided_run_id:
            provided_run_id = run.info.run_id
        print("Search is run_id ", run.info.run_id)
        experiment_id = run.info.experiment_id

        # Convert the mace model to pt for compatibility with LAMMPS
        build_model(provided_run_id, experiment_id, "gpu-delta.json", "models:/MACE Solid Hydrogen/2")


if __name__ == "__main__":
    run()
