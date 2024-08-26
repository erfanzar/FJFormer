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

"""Algorithms implemented in jaxpruner."""

from fjformer.jaxpruner.algorithms.global_pruners import (
	GlobalMagnitudePruning as GlobalMagnitudePruning,
)
from fjformer.jaxpruner.algorithms.global_pruners import (
	GlobalSaliencyPruning as GlobalSaliencyPruning,
)
from fjformer.jaxpruner.algorithms.pruners import MagnitudePruning as MagnitudePruning
from fjformer.jaxpruner.algorithms.pruners import RandomPruning as RandomPruning
from fjformer.jaxpruner.algorithms.pruners import SaliencyPruning as SaliencyPruning
from fjformer.jaxpruner.algorithms.sparse_trainers import SET as SET
from fjformer.jaxpruner.algorithms.sparse_trainers import RigL as RigL
from fjformer.jaxpruner.algorithms.sparse_trainers import (
	StaticRandomSparse as StaticRandomSparse,
)
from fjformer.jaxpruner.algorithms.ste import SteMagnitudePruning as SteMagnitudePruning
from fjformer.jaxpruner.algorithms.ste import SteRandomPruning as SteRandomPruning
