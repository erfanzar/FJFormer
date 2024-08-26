# Addapted from https://github.com/google-research/jaxpruner
# to make it available on pypi and use on Easydel.
# coding=utf-8
# Copyright 2024 Jaxpruner Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""APIs for jaxpruner."""

from fjformer.jaxpruner.algorithms import SET as SET
from fjformer.jaxpruner.algorithms import (
	GlobalMagnitudePruning as GlobalMagnitudePruning,
)
from fjformer.jaxpruner.algorithms import GlobalSaliencyPruning as GlobalSaliencyPruning
from fjformer.jaxpruner.algorithms import MagnitudePruning as MagnitudePruning
from fjformer.jaxpruner.algorithms import RandomPruning as RandomPruning
from fjformer.jaxpruner.algorithms import RigL as RigL
from fjformer.jaxpruner.algorithms import SaliencyPruning as SaliencyPruning
from fjformer.jaxpruner.algorithms import StaticRandomSparse as StaticRandomSparse
from fjformer.jaxpruner.algorithms import SteMagnitudePruning as SteMagnitudePruning
from fjformer.jaxpruner.algorithms import SteRandomPruning as SteRandomPruning
from fjformer.jaxpruner.api import ALGORITHM_REGISTRY as ALGORITHM_REGISTRY
from fjformer.jaxpruner.api import all_algorithm_names as all_algorithm_names
from fjformer.jaxpruner.api import (
	create_updater_from_config as create_updater_from_config,
)
from fjformer.jaxpruner.api import register_algorithm as register_algorithm
from fjformer.jaxpruner.base_updater import BaseUpdater as BaseUpdater
from fjformer.jaxpruner.base_updater import NoPruning as NoPruning
from fjformer.jaxpruner.base_updater import SparseState as SparseState
from fjformer.jaxpruner.base_updater import apply_mask as apply_mask
from fjformer.jaxpruner.sparsity_schedules import NoUpdateSchedule as NoUpdateSchedule
from fjformer.jaxpruner.sparsity_schedules import OneShotSchedule as OneShotSchedule
from fjformer.jaxpruner.sparsity_schedules import PeriodicSchedule as PeriodicSchedule
from fjformer.jaxpruner.sparsity_schedules import (
	PolynomialSchedule as PolynomialSchedule,
)
from fjformer.jaxpruner.sparsity_types import SparsityType as SparsityType
from fjformer.jaxpruner.utils import summarize_intersection as summarize_intersection
from fjformer.jaxpruner.utils import summarize_sparsity as summarize_sparsity
