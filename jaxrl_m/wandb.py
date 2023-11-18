"""WandB logging helpers.

Run setup_wandb(hyperparam_dict, ...) to initialize wandb logging.
See default_wandb_config() for a list of available configurations.

We recommend the following workflow (see examples/mujoco/d4rl_iql.py for a more full example):
    
    from ml_collections import config_flags
    from jaxrl_m.wandb import setup_wandb, default_wandb_config
    import wandb

    # This line allows us to change wandb config flags from the command line
    config_flags.DEFINE_config_dict('wandb', default_wandb_config(), lock_config=False)

    ...
    def main(argv):
        hyperparams = ...
        setup_wandb(hyperparams, **FLAGS.wandb)

        # Log metrics as you wish now
        wandb.log({'metric': 0.0}, step=0)


With the following setup, you may set wandb configurations from the command line, e.g.
    python main.py --wandb.project=my_project --wandb.group=my_group --wandb.offline
"""
import wandb

import tempfile
import absl.flags as flags
import ml_collections
from  ml_collections.config_dict import FieldReference
import datetime
import wandb
import time
import numpy as np

def setup_wandb(
    hyperparam_dict,
    entity=None,
    project="",
    group=None,
    name=None,
    unique_identifier="",
    **additional_init_kwargs,
):
    """
    Utility for setting up wandb logging (based on Young's simplesac):

    Arguments:
        - hyperparam_dict: dict of hyperparameters for experiment
        - project: str, wandb project name
        - entity: str, wandb entity name (default is your user)
        - group: str, Group name for wandb
        - name: str, Experiment name for wandb (formatted with FLAGS & hyperparameter_dict)
        - unique_identifier: str, Unique identifier for wandb (default is timestamp)
        - random_delay: float, Random delay for wandb.init (in seconds) to avoid collisions
        - additional_init_kwargs: dict, additional kwargs to pass to wandb.init
    Returns:
        - wandb.run

    """
    if group is not None and name is not None:
        experiment_id = f"{name}_{unique_identifier}"
    elif name is not None:
        experiment_id = f"{name}_{unique_identifier}"
    else:
        experiment_id = None

    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=hyperparam_dict,
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        id=experiment_id,
        name=name,
    )

    init_kwargs.update(additional_init_kwargs)
    run = wandb.init(**init_kwargs)
    wandb_config = dict(
        exp_prefix=group,
        exp_descriptor=name,
        experiment_id=experiment_id,
    )
    wandb.config.update(wandb_config)
    return run
