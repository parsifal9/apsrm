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

import os
import json
import pkgutil
from scipy.stats import betabinom
from math import floor



class BetaBernoulli:
    """A test with efficiency based on a bete Bernoulli distribution.

    You can find the data for this test in the file *pcrbetaparams.json* or
    simply view it :ref:`here<pcr-test-data>`.

    Instances of this class are callable.

    .. automethod:: __call__
    """

    params = json.loads(pkgutil.get_data(__name__, 'pcrbetaparams.json'))
    _alphas = params['alpha']
    _betas = params['beta']
    del params

    def __call__(self, person, period):
        """Test *person* in *period*.

        :return: *True* if *person* tests positive, *False* otherwise.
        :rtype: bool
        """
        # TODO: false positives?
        if not person.is_infected_by(period):
            return False

        time_since_infection = int(period) - person.period_infected

        # TODO: should this handle future infections (like it currently does)?
        if time_since_infection < 0 or time_since_infection >= len(self._alphas):
            return False

        return betabinom.rvs(
            1,
            self._alphas[time_since_infection],
            self._betas[time_since_infection],
            size = 1)[0] == 1.
