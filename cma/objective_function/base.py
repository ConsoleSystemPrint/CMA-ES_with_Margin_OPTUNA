#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
# public symbols
__all__ = ['ObjectiveFunction']




class ObjectiveFunction(object, metaclass=ABCMeta):
    """
    Abstract Class for objective function
    """

    minimization_problem = True

    def __init__(self, target_eval, max_eval):
        self.target_eval = target_eval
        self.max_eval = max_eval

        self.eval_count = 0
        self.best_eval = [np.inf] if self.minimization_problem else [-np.inf]

        self.is_better = (lambda x, y: x < y) if self.minimization_problem else (lambda x, y: x > y)
        self.is_better_eq = (lambda x, y: x <= y) if self.minimization_problem else (lambda x, y: x >= y)
        self.get_better = (
            lambda x, y: [np.minimum(xi, yi) for xi, yi in zip(x, y)]) if self.minimization_problem else (
            lambda x, y: [np.maximum(xi, yi) for xi, yi in zip(x, y)])
        self.get_best = (lambda evals: np.min(evals, axis=0).tolist()) if self.minimization_problem else (
            lambda evals: np.max(evals, axis=0).tolist())

    @abstractmethod
    def __call__(self, X):
        pass

    def clear(self):
        self.eval_count = 0
        self.best_eval = [np.inf] if self.minimization_problem else [-np.inf]

    def terminate_condition(self):
        if self.eval_count >= self.max_eval:
            return True
        return self.is_success()

    def is_success(self):
        if all(self.is_better_eq(be, te) for be, te in zip(self.best_eval, self.target_eval)):
            return True
        else:
            return False

    def verbose_display(self):
        return ' EvalCount: %d' % self.eval_count + ' BestEval: {}'.format(self.best_eval)

    @staticmethod
    def info_header():
        return ['EvalCount', 'BestEval']

    def info_list(self):
        return ['%d' % self.eval_count, '%e' % self.best_eval]

    def _update_best_eval(self, evals):
        self.best_eval = self.get_better(self.get_best(evals), self.best_eval)


class MultiObjectiveFunction(ObjectiveFunction):

    def __init__(self, target_eval, max_eval):
        super().__init__(target_eval, max_eval)

    @abstractmethod
    def __call__(self, X):
        pass


def objective(trial):
    x1 = trial.suggest_float('x1', 0, 10)
    x2 = trial.suggest_float('x2', 0, 10)

    obj1 = x1 ** 2 + x2 ** 2
    obj2 = (x1 - 5) ** 2 + (x2 - 5) ** 2

    return obj1, obj2


study = optuna.create_study(directions=["minimize", "minimize"], sampler=NSGAIISampler())
study.optimize(objective, n_trials=100)

for trial in study.best_trials:
    print(f'Trial ID: {trial.number}, Values: {trial.values}, Params: {trial.params}')