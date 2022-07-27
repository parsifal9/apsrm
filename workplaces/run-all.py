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
import pickle
from io import StringIO
import pandas as pd
from pandas.api.types import CategoricalDtype
import nbconvert as nbc
import nbformat as nbf

KERNEL = "python3"
OUTPUT_BASE_DIR = '../outputs'

def opath(base, filename):
    return os.path.join(OUTPUT_BASE_DIR, base, filename)

EVELEIGH_RESULTS = opath('eveleigh', 'all_results.pkl')
OFFICE_RESULTS = opath('office', 'all_results.pkl')

REPORT_NOTEBOOKS = [
    # for eveleigh
    'eveleigh/simulation.ipynb',

    # for office
    'office/simulation.ipynb',
    'office/office-ventilation-comparison.ipynb',
    'office/office-ventilation-comparison-single-day.ipynb',

    # for questacon
    'questacon/simulation.ipynb'
    ]



def generate_means_tables(
        results,
        R,
        baseline_column = 'BAU'):

    means = results[results.any_detected].groupby(['intervention']).mean()
    counts = results[results.any_detected].groupby('intervention').size()
    means_all = results.groupby(['intervention']).mean()

    means['relative_number_infected'] = \
        (1. - means_all['number_infected'] / means_all['number_infected'][baseline_column])
    means['number_infected'] = means_all['number_infected']
    means['any_detected'] = counts / R

    return means



def generate_combined_means_table(output_file_name):

    def load_infection_counts(file_name):
        with open(file_name, 'rb') as pkl:
            all_results, R = pickle.load(pkl)

        infection_counts = pd.concat([r[0] for r in all_results])
        dt = CategoricalDtype(categories=[r[1] for r in all_results], ordered=True)
        infection_counts['intervention'] = infection_counts['intervention'].astype(dt)

        return infection_counts, R

    if os.path.exists(EVELEIGH_RESULTS) and os.path.exists(OFFICE_RESULTS):
        eic, Re = load_infection_counts(EVELEIGH_RESULTS)
        oic, Ro = load_infection_counts(OFFICE_RESULTS)

    else:
        print('e res: {}'.format(EVELEIGH_RESULTS))
        print('e res: {}'.format(OFFICE_RESULTS))
        raise Exception('need all outputs to generate table')

    me = generate_means_tables(eic, Re)
    mo = generate_means_tables(oic, Ro)

    output = StringIO()

    # Write the header
    output.write('\n'.join([
        r'\begin{table}[H]',
        r'\rowcolors{2}{gray!25}{white}',
        r'\centering',
        r'\scriptsize',
        r"\caption{Average number of workers infected and average first period in which a case is detected for each intervention for each site considered. Here ``H'' refers to the hypothetical office, and ``E'' to the CSIRO Eveleigh office.} \label{tab:comparison}",
        r'\begin{tabular}{m{.075\textwidth}|m{.015\textwidth}m{.015\textwidth}|m{.03\textwidth}m{.03\textwidth}|m{.02\textwidth}m{.02\textwidth}|m{.02\textwidth}R{.02\textwidth}}',
        r'\toprule',
        r'{}&\multicolumn{2}{p{.03\textwidth}|}{Average Day Finished}&\multicolumn{2}{p{.06\textwidth}|}{Average Number of Cases}&\multicolumn{2}{p{.04\textwidth}|}{At Least One Case Detected (\%)}&\multicolumn{2}{p{.04\textwidth}}{Reduction in Number Infected c.f. BAU (\%)}\\',
        r'Intervention & H & E & H & E & H & E & H & E\\']) + '\n')

    formatters = {
        'period_finished': lambda v: str(round(v)),
        'number_infected': '{:0.2f}'.format,
        'any_detected': lambda v: str(round(100 * v)),
        'relative_number_infected': lambda v: str(round(100 * v))}

    # write the rows
    for i, (_, e), (_, o) in zip(me.index, me.iterrows(), mo.iterrows()):
        cols = ['period_finished', 'number_infected', 'any_detected', 'relative_number_infected']
        output.write(' & '.join(
            [i] +
            [' & '.join([formatters[col](o[col]), formatters[col](e[col])]) for col in cols]) +
            r'\\' + '\n')

    # write the footer
    output.write('\n'.join([r'\bottomrule', r'\end{tabular}', r'\end{table}']))

    with open(output_file_name, 'w') as tex:
        tex.write(output.getvalue())



def run_notebook(path):
    d, f = os.path.split(path)

    with open(path, "r") as f:
        nb = nbf.read(f, as_version=4)

    try:
        eb = nbc.preprocessors.ExecutePreprocessor(kernel_name=KERNEL)
        eb.preprocess(nb, resources={"metadata": {"path": d}})
    except:
        print('FAILURE: {}'.format(path))



for f in REPORT_NOTEBOOKS: run_notebook(f)
generate_combined_means_table(opath('.', 'comparison_table.tex'))
