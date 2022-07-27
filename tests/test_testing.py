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

import pytest as pt
from math import sqrt
from scipy.stats import (
    gamma,
    bernoulli,
    weibull_min as weibull)
from apsrm import Person, Pathogen
from apsrm.config import DEFAULT_PATHOGEN_DIEOFF_RATE
from apsrm._testing import BetaBernoulli

@pt.fixture
def pathogen():
    infectivity_dist    = gamma(3.420, 0, 1.338)
    incubation_dist     = weibull(3., 0, 7.2)
    shows_symptoms_dist = bernoulli(.5)
    return Pathogen(
        'delta',
        infectivity_function = lambda time_diff, person: infectivity_dist.pdf(time_diff),
        incubation_period_function = lambda person: incubation_dist.rvs(size=1),
        shows_symptoms_function = lambda person: shows_symptoms_dist.rvs(size=1)[0] == 1,
        is_honest_function = lambda person: True,
        dieoff_rate = DEFAULT_PATHOGEN_DIEOFF_RATE)

@pt.fixture
def person(pathogen):
    p = Person(42)
    p.infect(0, pathogen)
    return p

def test_load_beta_bernouli(person):
    bb = BetaBernoulli()
    n = 1000
    assert not bb(person, len(bb._alphas))
    assert not bb(person, -1)

    def do_period(period):
        a = bb._alphas[period]
        b = bb._betas[period]
        m = n*a / (a+b)
        v = n*a*b*(a + b + n) / ((a+b)*(a+b)*(a+b+1))
        s = sqrt(v)/n
        res = sum([bb(person, period) for i in range(n)])

        # will fail occasionally
        assert res == pt.approx(m, 5.*s)

    do_period(2)
    do_period(0)
    do_period(len(bb._alphas)-1)
