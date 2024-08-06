import mlflow
import os
import click

tracking_client = mlflow.tracking.MlflowClient()

def build_model(parent_run_id: str, experiment_id: int, backend_config: str, model_uri: str):
    p = mlflow.projects.run(
        uri=os.path.dirname(os.path.realpath(__file__)),
        entry_point="build-model",
        run_name="Build Model from MACE Model",
        parameters={
            "model_uri": model_uri,
            "parent_run_id": parent_run_id
        },
        experiment_id=experiment_id,
        synchronous=True,
        backend="slurm",
        backend_config=backend_config
    )

def gen_xyz(parent_run_id: str, experiment_id: int, pressure: int):
    p = mlflow.projects.run(
        uri=os.path.dirname(os.path.realpath(__file__)),
        entry_point="gen_xyz",
        run_name=f"Generate XYZ Files for Pressure {pressure}",
        parameters={
            "parent_run_id": parent_run_id,
            "pressure": pressure
        },
        experiment_id=experiment_id
    )

@click.command(help="Perform LAMMPS Simulation with Mace Model.")
@click.option("--model-uri", type=click.STRING,  help="Repository URI for Mace Model.", required=True)
@click.option("--pressures", type=click.STRING, help="Pressures to simulate.", required=True)
def run(model_uri: str, pressures: str):
    pressure_list = [int(pressure) for pressure in pressures.split(",")]
    provided_run_id = os.environ.get("MLFLOW_RUN_ID", None)
    with mlflow.start_run(run_id=provided_run_id) as run:
        if not provided_run_id:
            provided_run_id = run.info.run_id
        print("Search is run_id ", run.info.run_id)
        experiment_id = run.info.experiment_id

        # Convert the mace model to pt for compatibility with LAMMPS
        build_model(provided_run_id, experiment_id, "gpu-delta.json", model_uri=model_uri)

        # Create directories for each pressure and populate with position files
        for pressure in pressure_list:
            gen_xyz(provided_run_id, experiment_id, pressure)


if __name__ == "__main__":
    run()
