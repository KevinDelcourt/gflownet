import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tfloat, tlong
from torchtyping import TensorType
import gflownet.envs.nd_env.dimension
import ast


class NDEnv(GFlowNetEnv):
    """
    Custom N-Dimensional environment where each dimension i has its own number of possible values.

    At each step, the agent chooses one dimension that has not yet been assigned, and 
    picks one of its discrete values. The episode ends once all dimensions are chosen.

    Parameters
    ----------
    dim_profile : str or list of dimension sizes
        If str, must be a key in gflownet.envs.nd_env.dimension.dim_factories to generate dimensions.
        If list, should contain the sizes (number of possible values)

    """

    def __init__(
        self,
        dim_profile,
        **kwargs,
    ):        

        if isinstance(dim_profile, str):
            self.dimensions = gflownet.envs.nd_env.dimension.dim_factories[dim_profile]()
            self.dim_sizes = [len(dim.values) for dim in self.dimensions]
        else:
            self.dim_sizes = dim_profile
            self.dimensions = gflownet.envs.nd_env.dimension.make_dims(self.dim_sizes)

        self.dim_sizes_with_unassigned = [x + 1 for x in self.dim_sizes]
        self.n_dim = len(self.dim_sizes)
        self.cells = [dim.cells for dim in self.dimensions]
        self.source = [size for size in self.dim_sizes]  # means "unassigned"
        self.eos = (-1,-1)
        super().__init__(**kwargs)

    # ------------------------- ACTION SPACE -------------------------
    def get_action_space(self) -> List[Tuple[int, int]]:
        """All (dimension, value_index) pairs, plus EOS."""
        actions = []
        for dim, k in enumerate(self.dim_sizes):
            for v in range(k):
                actions.append((dim, v))
        actions.append(self.eos)
        return actions

    # ------------------------- STATE MANAGEMENT -------------------------

    def get_mask_invalid_actions_forward(
        self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """Invalid if dim already assigned, or EOS before completion."""
        if state is None:
            state = self.state
        if done is None:
            done = self.done
        if done:
            return [True] * self.policy_output_dim

        mask = []
        assigned_dims = {i for i, v in enumerate(state) if v != self.dim_sizes[i]}
        for act in self.action_space[:-1]:
            dim, _ = act
            mask.append(dim in assigned_dims)
        # EOS valid only if all dims assigned
        mask.append(not all(v != self.dim_sizes[i] for i, v in enumerate(state)))
        return mask

    def step(self, action: Tuple[int, int], skip_mask_check: bool = False):
        """Choose one dimension value."""
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )

        if not do_step:
            return self.state, action, False

        if action == self.eos:
            self.done = True
            self.n_actions += 1
            return self.state, action, True

        dim, val = action
        if self.state[dim] != self.dim_sizes[dim]:
            return self.state, action, False  # already assigned

        self.state[dim] = val
        self.n_actions += 1
        
        return self.state, action, True

    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        In continuous environments, get_parents() should return only the parent from
        which action leads to state.

        Args
        ----
        state : list
            Representation of a state

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : tuple
            Last action performed

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [tuple(self.eos)]

        parents, actions = [], []
        # Each assigned dim could have been the last one set
        for i, val in enumerate(state):
            if val != self.dim_sizes[i]:  # means assigned
                parent = state.copy()
                parent[i] = self.dim_sizes[i]  # mark as unassigned
                parents.append(parent)
                actions.append((i, val))
        

        return parents, actions

    # ------------------------- REPRESENTATIONS -------------------------
    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Converts integer states into flattened one-hot encoding for variable-length dims.
        Example:
            dim_sizes = [3, 4, 2]
            state = [1, 3, 0]
            => one-hot([1 of 3]) + one-hot([3 of 4]) + one-hot([0 of 2])
            -> vector length = 3 + 4 + 2 = 9
        """
        states = tlong(states, device=self.device)
        n_states = states.shape[0]

        # total number of cells across all dims (one extra state for unassigned value)
        total_len = sum(self.dim_sizes_with_unassigned) 

        # Preallocate
        states_policy = torch.zeros(
            (n_states, total_len), dtype=self.float, device=self.device
        )

        # cumulative offsets per dimension
        offsets = np.cumsum([0] + self.dim_sizes_with_unassigned[:-1])
        for i, (k, offset) in enumerate(zip(self.dim_sizes_with_unassigned, offsets)):
            idx_valid = (states[:, i] >= 0) & (states[:, i] < k)
            if torch.any(idx_valid):
                rows = torch.nonzero(idx_valid).flatten()
                cols = states[idx_valid, i] + offset
                states_policy[rows, cols] = 1.0

        return states_policy

    def states2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Converts integer states to continuous coordinates in [-1, 1] range for each dim.
        Works for variable-length dims (self.dim_sizes[i]).
        """
        states = tfloat(states, device=self.device, float_type=self.float)

        # Build one-hot encodings per dimension (variable length)
        states_policy = self.states2policy(states)

        # Split back into per-dimension blocks and weight by each dim's cell values
        start = 0
        coords = []
        for i, k in enumerate(self.dim_sizes):
            end = start + k
            onehot_block = states_policy[:, start:end]
            cell_tensor = torch.tensor(
                self.cells[i], device=self.device, dtype=self.float
            )
            coord = torch.matmul(onehot_block, cell_tensor)
            coords.append(coord)
            start = end
        output = torch.stack(coords, dim=1)  # shape [batch, n_dim]
        return output

    def _get_max_trajectory_length(self) -> int:
        """One step per dimension + EOS."""
        return self.n_dim + 1

    def get_all_terminating_states(self) -> List[List[int]]:
        grids = np.meshgrid(*[range(k) for k in self.dim_sizes])
        all_x = np.stack(grids).reshape((self.n_dim, -1)).T
        return all_x.tolist()

    def state2readable(self, state: Optional[List] = None):
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """
        state = self._get_state(state)
        readable = []
        for i, k in enumerate(state):
            readable.append(self.dimensions[i].values[k])
        return str(readable)

    def readable2state(self, readable):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        readable_list = ast.literal_eval(readable)
        state = []
        for i, val in enumerate(readable_list):
            state.append(self.dimensions[i].values.index(val))
        return state