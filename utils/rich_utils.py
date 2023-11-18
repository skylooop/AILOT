import rich
import rich.syntax
import rich.tree

from typing import *
from omegaconf import DictConfig, OmegaConf

def print_config_tree(cfg: DictConfig,
                      print_order: Sequence[str] = (
                          "algo",
                          "Env",
                          "GoalDS",
                          "logger"
                      )):
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    queue = []
    for field in print_order:
        if field in cfg:
            queue.append(field)
    for field in cfg:
        if field not in queue:
            queue.append(field)
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=False)
        else:
            branch_content = str(config_group)
        branch.add(rich.syntax.Syntax(branch_content, "yaml", line_numbers=True))
    rich.print(tree)
    