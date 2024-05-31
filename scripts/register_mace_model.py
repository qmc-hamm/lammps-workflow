import sys
import datetime
from pathlib import Path

import mlflow
import torch

training_command = """
python /home/ssjensen/mace_orig/mace/scripts/run_train.py \
       --name="M14-10-20-2023" \
       --train_file="data/train.xyz" \
       --test_file="data/test.xyz" \
       --E0s="average" \
       --model="MACE" \
       --hidden_irreps='128x0e + 128x1o' \
       --r_max=4.4 \
       --default_dtype='float32' \
       --batch_size=3 \
       --valid_batch_size=4 \
       --max_num_epochs=230 \
       --energy_weight=2200.0 \
       --forces_weight=10.0 \
       --swa \
       --start_swa=1200 \
       --ema \
       --ema_decay=0.99 \
       --amsgrad \
       --restart_latest \
       --device=cuda \
       --valid_fraction=0.05 \
       > train.out 2> train.err
"""


def extract_training_params(script):
    params = {}
    for line in script.split('  '):
        if '--' in line and '=' in line:
            key, value = line.split('=')
            key = key[2:].strip()
            value = value.strip()
            params[key] = value
    return params


def parse_log_line(line):
    """Parses a line from a log file and extracts the datetime, Epoch number, loss, RMSE per atom, and RMSE F.

  Args:
    line: A string representing a line from a log file.

  Returns:
    A tuple containing the datetime, Epoch number, loss, RMSE per atom, and RMSE F.
  """

    # Split the line into parts
    parts = line.split()

    # Parse the datetime
    datetime_str = parts[0] + " " + parts[1]
    datetime_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")

    # Parse the Epoch number
    epoch_str = parts[4]
    epoch_num = int(epoch_str.replace(":", ""))

    # Parse the loss
    loss_str = parts[5]
    loss = float(loss_str.replace("loss=", "").replace(",", ""))

    # Parse the RMSE per atom
    rmse_e_per_atom_str = parts[6]
    rmse_e_per_atom = float(
        rmse_e_per_atom_str.replace("RMSE_E_per_atom=", "").replace("meV", ""))

    # Parse the RMSE F
    rmse_f_str = parts[8]
    rmse_f = float(rmse_f_str.replace("RMSE_F=", "").replace("meV / A", ""))

    # Return the parsed values
    return datetime_obj, epoch_num, loss, rmse_e_per_atom, rmse_f


def extract_metrics(log_path: Path):
    with open(log_path) as f:
        for line in f:
            try:
                if 'Epoch' in line:
                    datetime_obj, epoch_num, loss, rmse_e_per_atom, rmse_f = parse_log_line(line)
                    mlflow.log_metric(
                        timestamp=int(datetime_obj.timestamp()),
                        key='loss',
                        value=loss,
                        step=epoch_num
                    )
                    mlflow.log_metric(
                        timestamp=int(datetime_obj.timestamp()),
                        key='rmse_e_per_atom',
                        value=rmse_e_per_atom,
                        step=epoch_num
                    )
                    mlflow.log_metric(
                        timestamp=int(datetime_obj.timestamp()),
                        key='rmse_f',
                        value=rmse_f,
                        step=epoch_num
                    )
            except ValueError as ve:
                print(f"Error parsing line: {line}, {ve}")
                pass

            if line.startswith("|    "):
                parts = line.split("|")
                dataset = parts[1].strip()
                rmse_e_per_atom = float(parts[2].strip())
                rmse_f = float(parts[3].strip())
                relative_rmse_f = float(parts[4].strip())
                mlflow.log_metric(f'{dataset}_rmse_e_per_atom', rmse_e_per_atom)
                mlflow.log_metric(f'{dataset}_rmse_f', rmse_f)
                mlflow.log_metric(f'{dataset}_relative_rmse_f', relative_rmse_f)


mlflow.set_tracking_uri("https://qmc-hamm.ml.software.ncsa.illinois.edu")
with mlflow.start_run(experiment_id="3", run_name="M18-01-09-2024") as run:
    mlflow.log_params(extract_training_params(training_command))
    extract_metrics(Path("ne1-cg6-freeze1/logs/M14-10-20-2023_run-123.log"))

    mlflow.pyfunc.log_model("mace.model",
        loader_module="mace.calculators",
        data_path="ne1-cg6-freeze1/M18-01-09-2024-freeze1.model")

    # model = torch.load("ne1-cg6-freeze1/M18-01-09-2024-freeze1.model")
    # mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifact("ne1-cg6-freeze1/logs/")
    mlflow.log_artifact("ne1-cg6-freeze1/results/")
    mlflow.log_artifact("ne1-cg6-freeze1/M18-01-09-2024-freeze1.model")
