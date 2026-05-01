from argparse import Namespace
import math
from torch.optim.lr_scheduler import OneCycleLR,StepLR, CosineAnnealingWarmRestarts
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import warnings

from .base import Coordinator, Node

optional = {"step_size": 100, "join_r": 1 , "max_lr": 0.01 , "T_0": 50, "scheduler": "StepLR", "upper_bound": 10,"infimum_lr": 1e-6}

def args_update(parser):
    parser.add_argument("--step_size", type=int, default=None)
    parser.add_argument("--join_r", type=float, default=None)
    parser.add_argument("--max_lr", type=float, default=None)
    parser.add_argument("--T_0", type=int, default=None)
    parser.add_argument("--upper_bound", type=int, default=None)
    parser.add_argument("--infimum_lr",type=float,default=None, help=f"Hyperbolic: Minimum learning rate")
    parser.add_argument("--scheduler", type=str, default=None)

class DFedAvg_Scheduler(Coordinator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_class = DFedAvg_Scheduler_Node

class DFedAvg_Scheduler_Node(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_scheduler()
        self.metrics['lr'] = []
    
    def get_scheduler(self):    
        if self.configs.scheduler == "StepLR":
            self.scheduler_class = StepLR(
                optimizer=self.optimizer, step_size=self.configs.step_size, gamma=0.5
            )
        elif self.configs.scheduler == "OCLR":
            self.scheduler_class = OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.configs.max_lr,
                total_steps=self.iterations * self.epochs,
            )
        elif self.configs.scheduler == "CAWR":
            self.scheduler_class = CosineAnnealingWarmRestarts(
                optimizer=self.optimizer, T_0=self.configs.T_0
            )
        elif self.configs.scheduler == "Hyperbolic":
            self.max_iter = self.iterations * self.epochs
            self.upper_bound = self.upper_bound * self.max_iter
            if not isinstance(self.upper_bound, int) or self.upper_bound <= 0:
                raise ValueError("upper_bound must be a positive integer.")
            if self.upper_bound <= self.max_iter:
                raise ValueError(
                    f"upper_bound ({self.upper_bound}) must be strictly greater than max_iter ({self.max_iter})."
                )
            N = float(self.max_iter)
            U = float(self.upper_bound)
            term0_squared_arg = (N / U) * (2.0 - N / U)
            if term0_squared_arg <= 0:
                raise ValueError(
                    f"Invalid parameters: N={N}, U={U} result in non-positive sqrt argument for initial term ({term0_squared_arg}). Check if U > N."
                )
            self._term0 = math.sqrt(term0_squared_arg)
            self._term0 = max(self._term0, 1e-12)
            self.scheduler_class = HyperbolicLR(
                optimizer=self.optimizer,
                configs=Namespace(
                    upper_bound=self.upper_bound,
                    max_epochs=self.max_iter,
                    infimum_lr=self.infimum_lr,  
                )
            )

    
    def train(self): 
        super().train()
        self.scheduler_class.step()
        self.logger.info(f'{self.optimizer.param_groups[0]["lr"] = }')
        self.metrics['lr'].append(self.optimizer.param_groups[0]["lr"])

class HyperbolicLR(_LRScheduler):
    """
    Paper: https://arxiv.org/abs/2407.15200
    Source: https://github.com/Axect/HyperbolicLR/blob/main/hyperbolic_lr.py
    """

    def __init__(self, optimizer: Optimizer, configs, last_epoch: int = -1):
        self.upper_bound = configs.upper_bound * configs.max_epochs
        self.max_iter = configs.max_epochs
        self.infimum_lr = configs.infimum_lr
        if not isinstance(self.upper_bound, int) or self.upper_bound <= 0:
            raise ValueError("upper_bound must be a positive integer.")
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if self.upper_bound <= self.max_iter:
            raise ValueError(
                f"upper_bound ({self.upper_bound}) must be strictly greater than max_iter ({self.max_iter})."
            )
        if not isinstance(self.infimum_lr, (float, int)) or self.infimum_lr < 0:
            raise ValueError("infimum_lr must be a non-negative number.")

        N = float(self.max_iter)
        U = float(self.upper_bound)
        term0_squared_arg = (N / U) * (2.0 - N / U)
        if term0_squared_arg <= 0:
            raise ValueError(
                f"Invalid parameters: N={N}, U={U} result in non-positive sqrt argument for initial term ({term0_squared_arg}). Check if U > N."
            )
        self._term0 = math.sqrt(term0_squared_arg)
        self._term0 = max(self._term0, 1e-12)
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)
        for base_lr in self.base_lrs:
            if self.infimum_lr >= base_lr:
                raise ValueError(
                    f"infimum_lr ({self.infimum_lr}) must be less than all base_lrs ({self.base_lrs})."
                )

    def get_lr(self):
        """Compute learning rate based on the current epoch (self.last_epoch)."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )


        current_iter = min(self.last_epoch, self.max_iter)

        N = float(self.max_iter)
        U = float(self.upper_bound)
        x = float(current_iter)

        if current_iter >= self.max_iter:
            scale_factor = 0.0
        else:
            termx_squared_arg = ((N - x) / U) * (2.0 - (N + x) / U)
            termx_squared_arg = max(0.0, termx_squared_arg)
            termx = math.sqrt(termx_squared_arg)
            scale_factor = termx / self._term0

        new_lrs = []
        for base_lr in self.base_lrs:
            delta_lr = base_lr - self.infimum_lr
            new_lr = self.infimum_lr + delta_lr * scale_factor
            new_lrs.append(max(self.infimum_lr, new_lr))

        return new_lrs