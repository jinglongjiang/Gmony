"""Backbone-independent initialization of inherited, non-temporal modules."""

import copy
import torch


def initialize_shared(policy, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for name, module in policy.named_children():
            if name == 'temporal_encoder':
                continue
            fresh = copy.deepcopy(module).cpu()
            for child in fresh.modules():
                if hasattr(child, 'reset_parameters'):
                    child.reset_parameters()
                elif hasattr(child, '_reset_parameters'):
                    child._reset_parameters()
            module.load_state_dict(fresh.state_dict())
