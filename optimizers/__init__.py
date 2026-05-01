import inspect
import sys

from torch.optim import SGD, Adam

# Automatically create a list of all classes imported in this file
OPTIMIZERS = [
    name for name, obj in sys.modules[__name__].__dict__.items() if inspect.isclass(obj)
]
print(f"{OPTIMIZERS = }")

from .FedProx import PerturbedGradientDescent
