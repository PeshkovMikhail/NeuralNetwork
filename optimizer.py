from collections import OrderedDict, defaultdict, abc as container_abcs
import torch
import warnings

from typing import Callable, Dict, List, Tuple

class Optimizer:
    def __init__(self, params, defaults):
        torch._C._log_api_usage_once("python.optimizer")
        self.defaults = defaults
        self._optimizer_step_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._optimizer_step_post_hooks: Dict[int, Callable] = OrderedDict()

        self._patch_step_function()

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

        # Allows _cuda_graph_capture_health_check to rig a poor man's TORCH_WARN_ONCE in python,
        # which I don't think exists
        # https://github.com/pytorch/pytorch/issues/72948
        self._warned_capturable_if_run_uncaptured = True


    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_optimizer_step_pre_hooks' not in self.__dict__:
            self._optimizer_step_pre_hooks = OrderedDict()
        if '_optimizer_step_post_hooks' not in self.__dict__:
            self._optimizer_step_post_hooks = OrderedDict()
        self._patch_step_function()  # To support multiprocessing pickle/unpickle
        self.defaults.setdefault('differentiable', False)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string


    def _optimizer_step_code(self):
        pass

    def _patch_step_function(self):
        self._zero_grad_profile_name = "Optimizer.zero_grad#{}.zero_grad".format(self.__class__.__name__)
        hooked = getattr(self.__class__.step, "hooked", None)
        if not hooked:
            self.__class__.step = self.profile_hook_step(self.__class__.step)
            self.__class__.step.hooked = True


    def zero_grad(self, set_to_none: bool = True):
        foreach = self.defaults.get('foreach', False)

        if not hasattr(self, "_zero_grad_profile_name"):
            self._patch_step_function()
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            if (not foreach or p.grad.is_sparse):
                                p.grad.zero_()
                            else:
                                per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)
            if foreach:
                for _, per_dtype_grads in per_device_and_dtype_grads.items():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)



    def step(self, closure):
        raise NotImplementedError


    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not self.defaults.get('differentiable', None) and not (param.is_leaf or param.retains_grad):
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
