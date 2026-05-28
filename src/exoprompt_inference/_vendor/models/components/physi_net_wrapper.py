from typing import Tuple, Any, Callable, Optional

import torch.nn as nn
from torch import Tensor
import torch


class PhysiNetWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        initial_w_physical: float = 0.99,
        initial_w_nn: float = 0.01,
        num_features: int = 1,
    ):
        super(PhysiNetWrapper, self).__init__()
        self.w_physical = nn.Parameter(Tensor([initial_w_physical] * num_features))
        self.w_nn = nn.Parameter(Tensor([initial_w_nn] * num_features))
        self.base_model = base_model

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        return super(PhysiNetWrapper, self).to(*args, **kwargs)

    def forward(
        self,
        x: Tensor | Any,
        y_physical: Tensor,
        # w_physical, w_nn, y_nn, y_physical
        custom_merge_ops: Optional[
            Callable[[nn.Parameter, nn.Parameter, Tensor, Tensor], Tuple]
        ] = None,
    ) -> Tuple[Tensor, Tensor]:
        if isinstance(x, tuple):
            y_nn = self.base_model(*x)
        else:
            y_nn = self.base_model(x)

        if custom_merge_ops is not None:
            (y_nn, y_combined), y_physical, y_nn_raw = custom_merge_ops(
                self.w_physical, self.w_nn, y_nn, y_physical
            )
            result = {
                "y_nn": y_nn,
                "y_nn_raw": y_nn_raw,
                "y_combined": y_combined,
                "y_physical": y_physical,
            }
        else:
            y_combined = (self.w_physical * y_physical) + (self.w_nn * y_nn)
            result = {
                "y_nn": y_nn,
                "y_combined": y_combined,
            }

        return result


if __name__ == "__main__":
    # TODO: @gsoykan - implement a mock case
    model = PhysiNetWrapper(base_model=None)

    # Example input data (x) and physics-based output (y_physical)
    x = torch.randn(100, 2)  # 2D input
    y_physical = torch.randn(100, 1)  # Corresponding physics-based outputs

    # Forward pass through PhysiNet
    y_nn, y_combined = model(x, y_physical)

    print(y_combined)
