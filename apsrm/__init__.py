# Copyright 2022 CSIRO
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

"""Core components for building spread models."""

import os
import re

from ._person import Person
from ._workplace import (
    Workplace,
    Gathering,
    GatheringGenerator,
    PeriodGenerator,
    EveryoneInfected)
from ._box import Box
from ._pathogen import Pathogen
from ._testing import BetaBernoulli
from .ventilation import (
    random_ventilation,
    balance_matrix,
    create_ventilation_system)
from ._vaccine import Vaccine

PCRTest = BetaBernoulli

with open(os.path.join(os.path.dirname(__file__), '..', 'version'), 'r') as vf:
    __version__ = vf.read().strip()

_v_regex = r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"

_version_info_match = re.match(_v_regex, __version__)

__version_info__ = tuple(int(_version_info_match.group(i)) for i in ('major',
    'minor', 'patch'))
