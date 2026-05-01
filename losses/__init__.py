import inspect
import sys

from torch.nn import CrossEntropyLoss as CEL

# Automatically create a list of all classes imported in this file
LOSSES = [
    name for name, obj in sys.modules[__name__].__dict__.items() if inspect.isclass(obj)
]
print(f"{LOSSES = }")
