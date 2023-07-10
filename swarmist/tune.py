from __future__ import annotations
from typing import Dict, Callable, cast, Union
from dataclasses import replace
from pymonad.either import Right, Left, Either
from swarmist.core.dictionary import (
    SearchStrategy,
    TuneStrategy,
    SearchResults,
    Initialization,
    Parameters,
    AutoInteger,
    Parameter,
    AutoParameter,
    TuneResults,
)
import optuna



class Tune:
    def __init__(
        self,
        strategy: TuneStrategy,
        search_callback: Callable[[SearchStrategy], Either[Exception, SearchResults]],
    ):
        self.strategy: TuneStrategy = strategy
        self.initialization: Initialization = strategy.strategy.initialization
        self.parameters: Dict[str, Union[Parameter, AutoParameter]] = strategy.strategy.parameters._parameters
        self.search_callback: Callable[
            [SearchStrategy], Either[Exception, SearchResults]
        ] = search_callback

    def _get_initialization(self, trial: optuna.Trial) -> Initialization:
        pop_size = self.initialization.population_size
        if isinstance(pop_size, AutoInteger):
            return replace(
                self.initialization,
                population_size=trial.suggest_int(
                    "population_size", pop_size.min, pop_size.max
                ),
            )
        return self.initialization

    def _get_parameters(self, trial: optuna.Trial) -> Parameters:
        params = Parameters()
        for k,v in self.parameters.items():
            if isinstance(v, AutoParameter):
                params.add(
                    k,
                    (
                        trial.suggest_int(v.name, v.value.min, v.value.max)
                        if isinstance(v.value, AutoInteger)
                        else trial.suggest_float(k, v.value.min, v.value.max)
                    ),
                )
            else:
                params.add(k, v.value)
        return params

    def objective(self, trial):
        initialization: Initialization = self._get_initialization(trial)
        parameters: Parameters = self._get_parameters(trial)
        st = replace(
            self.strategy.strategy, initialization=initialization, parameters=parameters
        )
        res = self.search_callback(st)
        if res.is_left():
            raise cast(Exception, res.monoid[0])
        return cast(SearchResults, res.value)[-1].fit

    def optimize(self, jobs: int = 1) -> SearchResults:
        study = optuna.create_study(direction="minimize")
        try:
            study.optimize(self.objective, n_trials=self.strategy.max_gen, n_jobs=jobs)
            trial = study.best_trial
            return Right(TuneResults(
                fit=trial.value,
                parameters=trial.params
            ))
        except Exception as e:
            return Left(e)
