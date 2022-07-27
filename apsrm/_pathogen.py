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

"""Contains the implementation of :py:class:`Pathogen`."""

import os
import json
import pkgutil
from math import exp
from functools import reduce
from typing import Callable
from numbers import Number
from ._person import Person

EMISSIONS = json.loads(pkgutil.get_data(__name__, 'emissions.json'))
BREATHING = EMISSIONS.pop('breathing')
EMISSIONS = EMISSIONS.pop('variants')
VARIANTS = [k for k in EMISSIONS.keys()]


class Pathogen:
    """A Pathogen.

    :param str name: The name of the pathogen. This must correspond to one of
        of the variants described in :py:data:`EMISSIONS`. At time of writing,
        this was one of *"wt"* or *"delta"* (see key *variants* in
        :ref:`emissions data<emissions-data>` for the possible options).

        If one wished to model some other variant with shedding rates that were
        proportional those of one of these variants, then

    :param Callable[[Number, Person], float] infectivity_function: A callable
        that takes the time since infection and an instance of
        :py:class:`Person` and returns the infectivity of that person the
        specified time since they became infected.

    :param Callable[[Person], float] incubation_period_function: A callable that
        takes an instance of :py:class:`Person` and returns the incubation period
        that person will experiences if they become infected.

    :param Callable[[Person], bool] shows_symptoms_function: A callable that
        takes an instance of :py:class:`Person` and returns a `True` if that
        person will show symptoms if they become infected and `False` otherwise.

    :param Callable[[Person], bool] is_honest_function: A callable that
        takes an instance of :py:class:`Person` and returns a `True` if that
        person will admit to showing symptoms if they become infected and do
        show symptoms and `False` otherwise.

    :param float dieoff_rate: The dieoff rate of the pathogen in the atmosphere.

    :param float gamma: A multiplier used in the exponential in the Wells-Riley
        equation (exposed on this class as :py:meth`probability_of_infection`),
        i.e. :math:`1-exp(-\\gamma x)`, where :math:`x` is the quanta of virus
        ingested.
    """

    def __init__(
            self,
            name,
            infectivity_function,
            incubation_period_function,
            shows_symptoms_function,
            is_honest_function,
            dieoff_rate,
            gamma = 1.):
        self.gamma = gamma
        self.name = name
        self.dieoff_rate = dieoff_rate
        self._infectivity_function = infectivity_function
        self._incubation_period_function = incubation_period_function
        self._shows_symptoms_function = shows_symptoms_function
        self._is_honest_function = is_honest_function

        infs = [infectivity_function(t, None) for t in range(31)]
        difs = [b-a for a, b in zip(infs[:-1], infs[1:])] + [0.]
        # assume the infectivity curve has a (maximum) turning point
        assert len(difs) > 2 and difs[-2] <= 0.
        self._last_time_significant = sum(
            d > 0. or i > 1e-4 for d, i in zip(difs, infs))


    @staticmethod
    def _get_weights(activity_weights, exhalation_rates):
        if isinstance(activity_weights, str):
            emission_rate = (exhalation_rates[activity_weights], 1.)

        else:
            emission_rate = reduce(
                lambda t, e: (t[0] + e[1] * exhalation_rates[e[0]], t[1] + e[1]),
                activity_weights.items() \
                    if isinstance(activity_weights, dict) \
                    else activity_weights, (0., 0.))

        return emission_rate


    @classmethod
    def breathing_rate(cls, person, period, activity_weights):
        total, sum_of_weights = cls._get_weights(activity_weights, BREATHING)
        return total / sum_of_weights


    def infectivity_at_time_since_infection(self,
            person: Person,
            period: Number,
            activity_weights: any) -> float:

        assert person.time_infected is not None

        if period < person.time_infected:
            # ... or throw an exception?
            return 0.

        total, sum_of_weights = self._get_weights(activity_weights, EMISSIONS[self.name])

        return total * self._infectivity_function(
                period - person.time_infected,
                person) / sum_of_weights


    def incubation_period(self, person: Person) -> float:
        """The incubation period that *person* will experience."""
        if self._shows_symptoms_function(person):
            return self._incubation_period_function(person)
        return float('inf')


    def is_honest(self, person: Person) -> bool:
        """Is *person* honest?

        :rtype: bool

        .. todo:: Move this as it is not an aspect of a pathogen... is it?
        """
        return self._is_honest_function(person)


    def probability_of_infection(self, ingested: Number) -> float:
        """Probability of getting infected given *ingested* quanta of virus has
        been ingested.

        This is the Wells-Riley equation: :math:`1-exp(-\\gamma \\times ingested)`.

        :param float ingested: Quanta of virus ingested.

        :rtype: float
        """
        pr_infection = 1. - exp(-self.gamma * ingested)

        # TODO: find a real solution for this.
        return max(0., pr_infection)


    def still_infectious_at(self, person: Person, period: Number) -> bool:
        """Is *person* still infectious in *period*?

        :param apsrm.Person person: The person to check.
        :param in period: The period to check for.

        :rtype: bool
        """
        pi = person.period_infected

        return pi is not None \
            and 0. <= (period - pi) <= self._last_time_significant



def concentration_at_time(t, S, G, V, Vf, pf, pd, C0):
    """Concentration at time *t*, given initial concentration of *C0*.

    :param t: Time in hours since the individual entered the box.

    :param S: Total quanta emission rate per unit time since time *t0* in quanta
        per hour. This is the sum of the emissions from all infected individuals
        in the box.

    :param G: The rate that fresh air flows into the box per unit time expressed
        as a fraction of the volume of the box. This is often denoted :math:`Gamma`,
        hence the use of G).

    :param V: The volume of the box in cubic meters.

    :param Vf: The volume of air passed through the internal air filter in
        cubic meters.

    :param pf: The efficiency of the internal air filter expressed as the
        proportion of particles removed.

    :param pd: The pathogen dieoff rate.

    :param C0: The concentration in the box at time 0.
    """
    G += Vf * pf / V + pd

    if G <= 0.:
        return C0 + t*S/V

    k = exp(-G*t)
    return k*C0 + (1-k)*S/(G*V)



def ingestion_by_time(t, S, G, V, Vf, pf, pd, f, p, C0, D0, t0):
    """How much is ingested by time.

    Assumes there is no dieoff once the pathogen has been ingested.

    :param t: Time in hours since the individual entered the box.

    :param S: Total quanta emission rate per unit time since time *t0* in
        quanta per hour.  This is the sum of the emissions from all infected
        individuals in the box.

    :param G: The rate that fresh air flows into the box per unit time expressed
        as a fraction of the volume of the box. This is often denoted :math:`Gamma`,
        hence the use of G).

    :param V: The volume of the box in cubic meters.

    :param Vf: The volume of air passed through the internal air filter in
        cubic meters.

    :param pf: The efficiency of the internal air filter expressed as the
        proportion of particles removed.

    :param pd: The pathogen dieoff rate.

    :param f: The efficiency of masks expressed as the proportion of particles
        removed. Note that this is assumed to apply to the individual for whom
        ingestion is being calculated. The effect of masks on shedding from
        infectious individuals is taken account of elsewhere.

    :param p: The breathing rate of the individual in cubic meters per hour.

    :param C0: The concentration in the box at time 0 in quanter per cubic
        meter.

    :param D0: The amount of virus previously ingested by the individual at the
        time they enter the box in quanta.

    :param t0: Time in hours at which the concentration was *C0*. *Q* and *S*
        are constant since that time.
    """
    G += Vf * pf / V + pd
    f = 1. - f

    if G <= 0.:
        return D0 + f*p*(t-t0)*(C0 + .5*S/V*(t + t0))

    Q  = G * V
    return D0 + t*f*p*S/Q - f*p * exp(-G*t0) * (1. - exp(-G*t)) * (S/(Q*G) - C0)
