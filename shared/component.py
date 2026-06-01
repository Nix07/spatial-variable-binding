import sys

sys.path.append("third_party/TransformerLens")
import transformer_lens as lens


class Component:
    """
    A wrapper class for a hookable component in a residual path in a transformer model.
    This extends the normal hooks functionality in transformer_lens by adding an optional
    head_idx parameter.
    """

    def __init__(self, hook_name, layer=None, head=None, neurons=None, position=None):
        self.hook_name = hook_name
        self.layer = layer
        self.head_idx = head
        self.neuron_indices = (
            tuple(neurons) if neurons is not None else None
        )  # Currently only supported for MLP neurons; Converted to tuple for hashability
        self.pos = position

    def __hash__(self):
        return hash(
            (self.hook_name, self.layer, self.head_idx, self.neuron_indices, self.pos)
        )

    def __eq__(self, other):
        # Compare two components by value and not by reference
        return (
            self.hook_name == other.hook_name
            and self.layer == other.layer
            and self.head_idx == other.head_idx
            and self.neuron_indices == other.neuron_indices
            and self.pos == other.pos
        )

    def valid_hook_name(self, layer=None) -> int:
        """
        Get a valid hook name for this component, which can be used to set a TransformerLens hook.
        This valid name is compatible with TransformerLens, thus does not contain any head / neuron information.

        Args:
            layer (int): The layer to get the valid hook name for. If None, the layer is taken from the component.
        """
        return lens.utils.get_act_name(name=self.hook_name, layer=layer or self.layer)

    @property
    def full_hook_name(self) -> str:
        """
        Get the full hook name (without regard to the layer) for visualization purposes.
        """
        full_hook_name = f"{self.hook_name}"
        if self.head_idx is not None:
            full_hook_name += f".head{self.head_idx}"
        if self.neuron_indices is not None:
            full_hook_name += ".specific_neurons"
        if self.pos is not None:
            full_hook_name += f".pos{self.pos}"
        return full_hook_name

    @property
    def is_mlp(self) -> bool:
        """
        Check if the component is an MLP component.
        """
        return "mlp" in self.hook_name

    @property
    def is_attn(self) -> bool:
        """
        Check if the component is an attention component.
        """
        valid_hook_name = self.valid_hook_name()
        for attn_hook_name in [
            "attn",
            "hook_q",
            "hook_k",
            "hook_v",
            "hook_z",
            "hook_pattern",
            "hook_result",
        ]:
            if attn_hook_name in valid_hook_name:
                return True
        return False

    @property
    def is_qkv(self) -> bool:
        """
        Checks if the component is a hook on either the Q/K/V tensors (post projection).
        """
        valid_hook_name = self.valid_hook_name()
        return (
            "attn.hook_q" in valid_hook_name
            or "attn.hook_k" in valid_hook_name
            or "attn.hook_v" in valid_hook_name
        )

    @property
    def is_resid(self) -> bool:
        """
        Check if the component is a residual stream component.
        """
        return "resid" in self.hook_name

    def __repr__(self) -> str:
        """
        Get the full hook name (with the layer).
        """
        return f"blocks.{self.layer}.{self.full_hook_name}"

    def __lt__(self, other) -> bool:
        return self.layer < other.layer or (
            self.layer == other.layer
            and self.head_idx is not None
            and other.head_idx is not None
            and self.head_idx < other.head_idx
        )
