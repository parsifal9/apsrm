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
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    gamma,
    bernoulli,
    weibull_min as weibull)
from ..config import (
    END_OF_PERIOD_TIME,
    DEFAULT_PATHOGEN_DIEOFF_RATE)
from .._pathogen import Pathogen
from .._workplace import EveryoneInfected
from .office import (
    add_ventilation_matrix,
    populate_workplace)



def run_simulation(
        workplace,
        pathogen,
        emissions_calculator,
        test=None,
        testing_fraction=None,
        post_period_functors=None):
    """A simple function for running a simulation.

    In this particular implementation, we infect a single individual at random,
    then run the model until either:

    - a case is detected, or
    - the risk to all non-infected individuals is below a given threshold for
      some period of time.
    """

    # infect a person
    workplace.infect_random_persons(pathogen)

    # This will stop when the first detection is made, or it is very unlikely
    # that anyone will be infected
    count = period = 0
    max_pr_of_infection = 1.
    any_detected = False
    all_infected = False
    risk_still_exists = True
    while not any_detected and risk_still_exists:

        workplace.reset(full=False)

        if not all_infected:
            try:
                max_pr_of_infection = workplace.run_period(
                    period,
                    pathogen,
                    emissions_calculator)

            except EveryoneInfected as e:
                all_infected = True
                max_pr_of_infection = 0.

        else:
            max_pr_of_infection = 0.

        risk_still_exists = max_pr_of_infection > 1e-4 or any(
            pathogen.still_infectious_at(person, period) for person in \
                workplace.infected_schedules.keys())

        if test is not None:
            any_detected = workplace.run_testing(
                test,
                period,
                only_symptomatic = testing_fraction is None,
                proportion_to_test = testing_fraction)

        count = workplace.count_infected()
        if post_period_functors is not None:
            for f in post_period_functors:
                f(period, workplace)

        period += 1

    return {
        'period_finished': period,
        'number_infected': count,
        'any_detected': any_detected}



_infectivity_dist    = gamma(3.420, 0, 1.338)
_incubation_dist     = weibull(3., 0, 7.2)
_shows_symptoms_dist = bernoulli(.5)
#_honest_dist = bernoulli(.9)
_infectivity_binned  = np.diff(_infectivity_dist.cdf(np.arange(25))) / _infectivity_dist.cdf(END_OF_PERIOD_TIME)
def _infectivity(time_diff, person):
    #the following alternative took 70% or so of total run time!
    #lambda time_diff, person: _infectivity_dist.pdf(time_diff),
    if time_diff < 0. or time_diff >= len(_infectivity_binned):
        return 0.
    return _infectivity_binned[floor(time_diff)]



def create_pathogen(
        variant,
        gamma=1.,
        pathogen_dieoff_rate=DEFAULT_PATHOGEN_DIEOFF_RATE):

    return Pathogen(
        name = variant,
        infectivity_function = _infectivity,
        incubation_period_function = lambda person: _incubation_dist.rvs(size=1)[0],
        shows_symptoms_function = lambda person: _shows_symptoms_dist.rvs(size=1)[0] == 1,
        is_honest_function = lambda person: True,#_honest_dist.rvs(size=1)[0] == 1,
        dieoff_rate = pathogen_dieoff_rate,
        gamma = gamma)



class EmissionsCalculator:
    def __init__(self, pathogen, activity_weights, mask_wearing):
        self._pathogen = pathogen
        self._activity_weights = activity_weights
        self._mask_wearing = mask_wearing

    def emissions(self, time, person, interval, gathering):
        weights = self._activity_weights[(person.role, interval.box.use)]
        if callable(weights): weights = weights(person, gathering)
        filtering = self.shedding_filtering_in_box(person, interval)
        return (1. - filtering) * self._pathogen.infectivity_at_time_since_infection(
            person, time, weights)

    def breathing_rate(self, time, person, interval, gathering):
        weights = self._activity_weights[(person.role, interval.box.use)]
        if callable(weights): weights = weights(person, gathering)
        return self._pathogen.breathing_rate(
            person, time, weights)

    def person_wears_mask_in(self, person, interval):
        if hasattr(interval, 'wearing_mask'):
            return interval.wearing_mask
        return self._mask_wearing.get((person.role, interval.box.use), True)

    def shedding_filtering_in_box(self, person, interval):
        return person.shedding_filter_efficiency \
            if self.person_wears_mask_in(person, interval) \
            else 0.

    def ingestion_filtering_in_box(self, person, interval):
        return person.ingestion_filter_efficiency \
            if self.person_wears_mask_in(person, interval) \
            else 0.



# TODO: Remove defaults and put them in wrappers?
def create_workplace(
    workplace_loader,
    box_types,
    worker_types,
    hvac_box_type = None,
    frac_of_max_occupancy = .7,
    receptionist_count = 1,
    external_acph = 1.,
    hvac_return_filtering_efficiency = 0.,
    air_cleaner_filtering_efficiency = 0.,
    air_cleaner_filtering_volume = 0.,
    air_cleaner_box_types = None,
    proportion_leaving_for_lunch = None,
    hvac_acph = None,
    inter_box_acph = None,
    workplace = None):

    add_hvac = (hvac_acph is not None) and (inter_box_acph is not None)

    if workplace is None:
        populate = True
        workplace, ventilation_matrix = workplace_loader(add_hvac)
    else:
        populate = False
        _, ventilation_matrix = workplace_loader(add_hvac)

    add_ventilation_matrix(
        workplace = workplace,
        external_acph = external_acph,
        air_cleaner_filtering_efficiency = air_cleaner_filtering_efficiency,
        air_cleaner_filtering_volume = air_cleaner_filtering_volume,
        air_cleaner_box_types = air_cleaner_box_types,
        ventilation_matrix = ventilation_matrix,
        hvac_acph = hvac_acph,
        inter_box_acph = inter_box_acph,
        hvac_box_type = hvac_box_type,
        hvac_return_filtering_efficiency = hvac_return_filtering_efficiency)

    if populate:
        populate_workplace(
            workplace = workplace,
            box_types = box_types,
            worker_types = worker_types,
            frac_of_max_occupancy = frac_of_max_occupancy,
            receptionist_count = receptionist_count,
            proportion_leaving_for_lunch = proportion_leaving_for_lunch)

    return workplace



def generate_means_tables(
        results,
        R,
        caption,
        label,
        baseline_column = 'BAU'):

    means = results[results.any_detected].groupby(['intervention']).mean()
    counts = results[results.any_detected].groupby('intervention').size()
    means_all = results.groupby(['intervention']).mean()

    means['relative_number_infected'] = \
        (1. - means_all['number_infected'] / means_all['number_infected'][baseline_column])
    means['number_infected'] = means_all['number_infected']
    means['any_detected'] = counts / R

    means_latex = means.to_latex(
        header=[
            'Average Period Finished (days)',
            'Average Number Infected (workers)',
            'Fraction of Simulations Where At Least One Case was Detected (%)',
            'Number Infected Reduction From BAU (%)'],
        formatters = {
            'period_finished': lambda v: str(round(v)),
            'number_infected': '{:0.2f}'.format,
            'any_detected': lambda v: str(round(100 * v)),
            'relative_number_infected': lambda v: str(round(100 * v))},
        column_format=r'L{.19\textwidth}R{.17\textwidth}R{.17\textwidth}R{.17\textwidth}R{.17\textwidth}',
        label=label,
        position='htbp')

    return means, means_latex



def _plot_histograms(
        d,
        n_plots,
        image_base_path,
        image_prefix,
        column,
        bins,
        title,
        by='intervention',
        density=True,
        color='gray',
        image_format='pdf',
        max_n_col = 4):

    def nrc(n):
        if n in (5, 6):
            return 2, 3

        if n == 9:
            return 3, 3

        n_last_row = n % max_n_col
        n_row = floor(n / max_n_col)

        if n_last_row != 0:
            n_row += 1

        if n_row == 1:
            return 1, n
        else:
            return n_row, max_n_col

    fig_dim = nrc(n_plots)

    axarr = d.hist(
        column=column,
        by=by,
        bins=bins,
        sharey=True,
        layout=fig_dim,
        figsize=(fig_dim[1] * 14. / 6., fig_dim[0] * (3. if fig_dim[0] == 1 else (.75 * 3.75))),
        align='left',
        color=color,
        density=density)

    ylab_prefix = 'Proportion' if density else 'Number'
    for i, ax in enumerate(axarr.flatten()):
        if i == 0:
            ax.set_ylabel('{} of Simulations'.format(ylab_prefix))
            ax.set_xlabel(title)
    plt.tight_layout()
    plt.savefig(os.path.join(image_base_path, '{}_{}.{}'.format(image_prefix, column, image_format)))
    plt.show()



def plot_histograms(
        infection_counts,
        image_base_path,
        image_prefix):

    n_plots = len(np.unique(infection_counts.intervention))
    max_infected = np.max(infection_counts['number_infected'].values)
    max_period = np.max(infection_counts['period_finished'].values)

    _plot_histograms(
        infection_counts,
        n_plots,
        image_base_path,
        image_prefix,
        column='number_infected',
        bins=np.arange(max_infected + 2),
        title='Workers Infected')

    _plot_histograms(
        infection_counts[infection_counts.any_detected],
        n_plots,
        image_base_path,
        image_prefix,
        column='period_finished',
        bins=np.arange(max_period + 2),
        title='Days Until First Detection')
