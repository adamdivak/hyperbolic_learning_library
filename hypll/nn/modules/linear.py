from torch.nn import Module

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match


class HLinear(Module):
    """Poincare fully connected linear layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: Manifold,
        bias: bool = True,
        init_scale: float = 1.0,
        init_std: float = 0.01,
        init_type: str = 'normal',
    ) -> None:
        super(HLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.has_bias = bias

        self.init_scale = init_scale
        self.init_std = init_std
        self.init_type = init_type

        # TODO: torch stores weights transposed supposedly due to efficiency
        # https://discuss.pytorch.org/t/why-does-the-linear-module-seems-to-do-unnecessary-transposing/6277/7
        # We may want to do the same
        self.z, self.bias = self.manifold.construct_dl_parameters(
            in_features=in_features, out_features=out_features, bias=self.has_bias
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.manifold.reset_parameters(
            weight=self.z, 
            bias=self.bias, 
            init_type=self.init_type,
            scale=self.init_scale,
            init_std=self.init_std,
        )

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=x)
        check_if_man_dims_match(layer=self, man_dim=-1, input=x)
        return self.manifold.fully_connected(x=x, z=self.z, bias=self.bias)

class HMlr(Module):
    """Poincare Multinomial Logistic Regression layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: Manifold,
        bias: bool = True,
        init_scale: float = 1.0,
        init_std: float = 0.01,
        init_type: str = 'normal',
    ) -> None:
        super(HMlr, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self.has_bias = bias

        self.init_scale = init_scale
        self.init_std = init_std
        self.init_type = init_type

        # TODO: torch stores weights transposed supposedly due to efficiency
        # https://discuss.pytorch.org/t/why-does-the-linear-module-seems-to-do-unnecessary-transposing/6277/7
        # We may want to do the same
        self.z, self.bias = self.manifold.construct_dl_parameters(
            in_features=in_features, out_features=out_features, bias=self.has_bias
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.manifold.reset_parameters(
            weight=self.z, 
            bias=self.bias, 
            init_type=self.init_type,
            scale=self.init_scale,
            init_std=self.init_std,
        )

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        check_if_manifolds_match(layer=self, input=x)

        original_x_data_shape = x.tensor.shape
        original_x_man_dim = x.man_dim
        input_ndim = x.tensor.ndim

        # Overly verbose code to handle 3d input to MLR correctly
        # Reshape if input is >2D and its man_dim is the last dimension.
        # The core self.manifold.mlr (via poincare_hyperplane_dists) expects
        # the input ManifoldTensor to have man_dim = 1 (i.e., a 2D tensor [N, features]).
        if input_ndim > 2 and original_x_man_dim == (input_ndim - 1):
            is_reshaped = True
            
            # Flatten all dimensions before the manifold dimension (the last one).
            # e.g., (B, S, E) with original_x_man_dim=2 -> results in (B*S, E).
            # The new man_dim for this (N, E) tensor will be 1.
            
            # Calculate the size of the first dimension after flattening (e.g., B*S)
            num_prefix_elements = 1
            for i in range(original_x_man_dim): # Iterate dims before the manifold dim
                num_prefix_elements *= original_x_data_shape[i]
            
            # Size of the manifold dimension itself (e.g., E)
            man_dim_size = original_x_data_shape[original_x_man_dim]
            
            # Reshape the data tensor to 2D: (num_prefix_elements, man_dim_size)
            x_data_reshaped = x.tensor.reshape(num_prefix_elements, man_dim_size)
            
            # Create a new ManifoldTensor for the reshaped data.
            # Its man_dim is 1 (0-indexed, so the second dimension of the [N, E] tensor).
            x_for_mlr = ManifoldTensor(
                data=x_data_reshaped,
                manifold=x.manifold, # Propagate manifold type
                man_dim=1 
            )
        else:
            # Input is already 2D, or its man_dim is not the last one (unexpected).
            # If it's 2D, its man_dim should ideally be 1.
            # If not, self.manifold.mlr will likely raise the same error as before.
            is_reshaped = False
            x_for_mlr = x

        # Call the core MLR logic. self.manifold.mlr should return a ManifoldTensor.
        output_mlr = self.manifold.mlr(x=x_for_mlr, z=self.z, bias=self.bias)

        # Reshape output back if the input was reshaped
        if is_reshaped:
            # output_mlr.data is expected to be 2D: (num_prefix_elements, self.out_features)
            # Restore the original prefix dimensions.
            # e.g., if original_x_data_shape was (B, S, E),
            # output_mlr.data is (B*S, out_features).
            # We want final output data shape (B, S, out_features).
            
            # Construct the target shape for the output data:
            # (original_x_data_shape[0], ..., original_x_data_shape[original_x_man_dim-1], self.out_features)
            final_output_data_shape_list = []
            for i in range(original_x_man_dim): # These are the B, S, ... dimensions
                final_output_data_shape_list.append(original_x_data_shape[i])
            final_output_data_shape_list.append(self.out_features) # The new feature dimension size
            
            final_output_data_reshaped = output_mlr.tensor.reshape(*final_output_data_shape_list)
            
            # The man_dim of the final output tensor will be its last dimension's index.
            # e.g., for (B, S, out_features), it's 2 (0-indexed for 3 dims).
            final_output_man_dim = final_output_data_reshaped.ndim - 1
            
            final_output = ManifoldTensor(
                data=final_output_data_reshaped,
                manifold=output_mlr.manifold, # Propagate manifold type
                man_dim=final_output_man_dim
            )
        else:
            final_output = output_mlr
            
        return final_output
