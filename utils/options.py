import argparse
import json
import os

from rich import box
from rich.console import Console
from rich.table import Table
from rich.terminal_theme import MONOKAI

from dataset_factory import DATA_PARTITIONS, DATASETS
from losses import LOSSES
from models import MODELS
from optimizers import OPTIMIZERS
from strategies import STRATEGIES
from topologies import TOPOLOGIES

from .general import increment_path


class Options:
    def __init__(self, root):
        self.root = root

    def parse_options(self):
        parser = argparse.ArgumentParser()
        # general
        parser.add_argument("--seed", type=int, default=941)
        parser.add_argument(
            "--times", type=int, default=1, help="number of times to run the experiment"
        )
        parser.add_argument(
            "--prev", type=int, default=0, help="Previous Running times"
        )
        parser.add_argument(
            "--device", type=str, default="cuda", choices=["cpu", "cuda"]
        )
        parser.add_argument("--device_id", type=str, default="0")
        parser.add_argument("--parallel", type=bool, default=False, help="parallel execution")
        parser.add_argument(
            "--workers", type=int, default=1, help="number of workers for dataloader"
        )
        parser.add_argument("--save_local_model", action="store_true", default=None)
        parser.add_argument(
            "--topology", type=str, default="FullyConnected", choices=TOPOLOGIES
        )
        parser.add_argument("--k", type=int, default=7)
        # save path
        parser.add_argument(
            "--project",
            type=str,
            default=os.path.join(self.root, "runs"),
            help="project name",
        )
        parser.add_argument(
            "--name", type=str, default="exp", help="name of this experiment"
        )
        parser.add_argument("--sep", type=str, default="", help="separator for name")
        parser.add_argument(
            "--convergence_targets",
            type=str,
            default=None,
            help="target accuracy to monitor convergence",
        )

        # dataset
        parser.add_argument(
            "--dataset",
            type=str,
            default="cifar100",
            help="name of dataset",
            choices=DATASETS,
        )
        parser.add_argument(
            "--num_nodes", type=int, default=20, help="number of nodes"
        )
        parser.add_argument("--datatest", type=bool, default=True, help="full datatest")
        parser.add_argument(
            "--train_ratio", type=float, default=0.75, help="train ratio"
        )
        parser.add_argument("--batch_size", type=int, default=10, help="batch size")
        parser.add_argument("--alpha", type=float, default=0.1, help="alpha")
        parser.add_argument("--iid", action="store_true", default=False, help="niid")
        parser.add_argument(
            "--balance", action="store_true", default=False, help="balance"
        )
        parser.add_argument(
            "--partition",
            type=str,
            default="dir",
            help="data partition",
            choices=DATA_PARTITIONS,
        )
        parser.add_argument(
            "--eval_method",
            type=str,
            default="generalization",
            choices=["generalization", "personalization"],
        )
        parser.add_argument(
            "--class_per_client",
            type=int,
            default=None,
            help="number of classes per client",
        )
        parser.add_argument(
            "--plot_ylabel_step", type=int, default=None, help="plot ylabel step"
        )

        # server
        parser.add_argument(
            "--framework", type=str, default="FedAvg", choices=STRATEGIES
        )
        parser.add_argument("--model", type=str, default="FedAvgCNN", choices=MODELS)
        parser.add_argument("--iterations", type=int, default=2000)
        parser.add_argument(
            "--patience", type=int, default=None, help="Patience for early stopping"
        )
        parser.add_argument(
            "--join_ratio", type=float, default=1.0, help="Ratio of nodes per round"
        )
        parser.add_argument(
            "--random_join_ratio",
            type=bool,
            default=False,
            help="Random ratio of nodes per round",
        )
        parser.add_argument(
            "--eval_gap", type=int, default=1, help="Rounds gap for evaluation"
        )
        parser.add_argument(
            "--decoupling",
            action="store_true",
            default=False,
            help="model decoupling, split model into two parts: base (feature extractor) and head",
        )
        parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

        # client
        parser.add_argument(
            "--optimizer",
            type=str,
            default="SGD",
            help="optimizer name",
            choices=OPTIMIZERS,
        )
        parser.add_argument(
            "--learning_rate", type=float, default=0.005, help="Local learning rate"
        )
        parser.add_argument(
            "--in_channels", type=int, default=3, help="Number of input channels"
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=1,
            help="Multiple update steps in one local epoch.",
        )
        parser.add_argument(
            "--loss", type=str, default="CEL", help="loss function", choices=LOSSES
        )
        parser.add_argument(
            "--max_seq_len", type=int, default=25, help="Maximum sequence length"
        )
        parser.add_argument(
            "--dim", type=int, default=1024, help="Dimension for FedAvgCNN"
        )

        getattr(__import__("strategies"), "apply_args_update")(parser)

        self.args = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.device_id
        return self

    def _fix_framework_specific_param(self):
        optional = getattr(__import__("strategies"), "optional")
        self.update_if_none(params=optional.get(self.args.framework, {}))

        compulsory = getattr(__import__("strategies"), "compulsory")
        self.update_args(params=compulsory.get(self.args.framework, {}))
        pass

    def update_args(self, params: dict):
        """
        Update self.args with a given dictionary of parameters.

        Args:
            params (dict): Dictionary containing parameter updates.
        """
        for key, value in params.items():
            self.update_arg(key, value)

    def update_if_none(self, params: dict):
        """
        Update self.args with a given dictionary of parameters only if they are None.

        Args:
            params (dict): Dictionary containing parameter updates.
        """
        for key, value in params.items():
            if getattr(self.args, key) is None or key not in self.args:
                self.update_arg(key, value)

    def update_arg(self, name, value):
        self.args.__dict__[name] = value

    # def _fix_topology(self):
    #     compulsory = {
    #         'CenFL': {'topology': 'Star'},
    #     }
    #     self.update_args(params=compulsory.get(self.args.framework, {}))

    def _fix_save_path(self):
        path = increment_path(
            os.path.join(self.args.project, self.args.name),
            exist_ok=False,
            sep=self.args.sep,
        )
        self.update_arg("save_path", path)
        if not os.path.exists(path):
            os.makedirs(path)
    


    def _fix_dataset(self):
        update_dict = {
            "cifar10": {
                "class_per_client": 2,
                "plot_ylabel_step": 1,
                "in_channels": 3,  # CIFAR10 has 3 color channels
                "dim": 1024        # Add dim for FedAvgCNN
            },
            "cifar100": {
                "class_per_client": 10,
                "plot_ylabel_step": 20,
                "in_channels": 3,  # CIFAR100 has 3 color channels
                "dim": 1024        # Add dim for FedAvgCNN
            },
            "tinyimagenet": {
                "class_per_client": 10,
                "plot_ylabel_step": 40,
                "in_channels": 3,  # TinyImageNet has 3 color channels
                "dim": 4096        # 512 * 8 * 8 if using a ResNet-like pooling
            },
        }
        self.update_if_none(params=update_dict[self.args.dataset])

    def _fix_device(self):
        import torch

        if self.args.device == "cuda" and not torch.cuda.is_available():
            print("cuda is not available. Using cpu instead.")
            self.update_arg("device", "cpu")

    def _fix_model(self):
        if self.args.model in ["FedAvgCNN"]:
            if "cifar10" in self.args.dataset.lower():
                dim = 1600
                in_features = 3
            else:
                dim = 10816
                in_features = 3
            self.update_args({"dim": dim, "in_features": in_features})

    def fix_args(self):
        self._fix_save_path()
        self._fix_device()
        self._fix_framework_specific_param()
        self._fix_model()
        self._fix_dataset()
        # self._fix_topology()

        return self

    def save(self):
        """
        Saves all values in self.args to a file.
        """
        with open(os.path.join(self.args.save_path, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)

    def display(self):
        """
        Displays all values in self.args using rich package.
        """

        table = Table(title="Experiment Arguments", box=box.ROUNDED)
        table.add_column("Argument", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for arg in vars(self.args):
            table.add_row(arg, str(getattr(self.args, arg)))

        console = Console(record=True)
        console.print(table)
        console.save_svg(
            os.path.join(self.args.save_path, "configs.svg"), theme=MONOKAI
        )
