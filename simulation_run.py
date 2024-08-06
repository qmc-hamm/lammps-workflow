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

def run_lammps(parent_run_id: str, experiment_id: int, backend_config: str,
    pressure: int, temperature: int, nbeads: int, iterations: int, iterations_per_job: int):

    assert iterations % iterations_per_job == 0 # Ensure we can divide the iterations into evenly sized jobs
    num_jobs = iterations // iterations_per_job

    p = mlflow.projects.run(
        uri=os.path.dirname(os.path.realpath(__file__)),
        entry_point="simulation",
        run_name=f"Run LAMMPS Simulation for Pressure {pressure} and Temperature {temperature}",
        parameters={
            "parent_run_id": parent_run_id,
            "pressure": pressure,
            "temperature": temperature,
            "nbeads": nbeads,
            "iterations": iterations_per_job,
            "sequential_workers": num_jobs,
            "case": "QMC"
        },
        experiment_id=experiment_id,
        synchronous=False,
        backend="slurm",
        backend_config=backend_config
    )
@click.command(help="Perform LAMMPS Simulation with Mace Model.")
@click.option("--model-uri", type=click.STRING,  help="Repository URI for Mace Model.", required=True)
@click.option("--pressures", type=click.STRING, help="Pressures to simulate.", required=True)
@click.option("--temperatures", type=click.STRING, help="Temperatures to simulate.", required=True)
@click.option("--nbeads", type=click.INT, help="Number of beads.", required=False, default=16)
@click.option("--iterations", type=click.INT, help="Total number of iterations.", required=False, default=9000)
@click.option("--iterations-per-job", type=click.INT, help="Number of iterations that can be achieved in a single job", required=False, default=1000)
def run(model_uri: str, pressures: str, temperatures: str, nbeads: int,
    iterations, iterations_per_job):
    pressure_list = [int(pressure) for pressure in pressures.split(",")]
    temperature_list = [int(temperature) for temperature in temperatures.split(",")]
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

        # Now run LAMMPS simulations over each pressure and temperature
        for pressure in pressure_list:
            for temperature in temperature_list:
                run_lammps(provided_run_id, experiment_id, "cpu-delta.json",
                    pressure, temperature, nbeads, iterations, iterations_per_job)



if __name__ == "__main__":
    run()
