#!/bin/env python
#
# Author: Martin Kocour (BUT)

import os
import torch
import math

from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import parameters_to_vector


def get_grads(params):
    _grads = [
        param.grad.detach().flatten()
        for param in params
        if param.grad is not None
    ]
    return torch.cat(_grads)


class GeneralMetricWritter(object):
    def __init__(self, model, exp_name=None):
        log_dir = None if exp_name is None else os.path.join("runs", exp_name)
        self._writter = SummaryWriter(log_dir=log_dir)
        self._model = model
        self._step = 0

    def log_metrics(self, loss, step=None, train=True):
        if step is None:
            step = self._step
            self._step += 1

        model = self._model
        if train:
            grad_norm = get_grads(model.parameters()).norm()
            self._writter.add_scalar("grad_norm", grad_norm, step)
            self._writter.add_scalar("Loss/train", loss, step)
        else:
            model_norm = parameters_to_vector(model.parameters()).norm()
            self._writter.add_scalar("model_norm", model_norm, step)
            self._writter.add_scalar("Loss/dev", loss, step)

        if step % 100 == 0:
            self._writter.flush()
        return step

    def log_hparams(self, hparam_dict, metric_dict, **kwargs):
        metric_dict["hparam/nparams"] = sum([p.numel() for p in self._model.parameters()])
        self._writter.add_hparams(hparam_dict, metric_dict, **kwargs)

    def close(self):
        self._writter.close()


class LMMetricWritter(GeneralMetricWritter):
    def log_metrics(self, loss, step=None, train=True):
        step = super().log_metrics(loss, step, train)
        if train:
            self._writter.add_scalar("PPL/train", math.exp(loss), step)
        else:
            self._writter.add_scalar("PPL/dev", math.exp(loss), step)
        return step

