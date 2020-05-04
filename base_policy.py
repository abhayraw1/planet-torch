from torch import nn


class Policy(nn.Module):
    """Abstract base class for policy.
    Should be inherited by all the policies
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def poll(self, *args):
        """To be implemented in the child class
        should return action tensor
        """
        return self(*args)
