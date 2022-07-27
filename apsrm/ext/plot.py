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

import re
from math import floor
from functools import reduce
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from ..config import DEFAULT_PATHOGEN_DIEOFF_RATE

ONE_MINUTE = 1./60.



def _safe_max(*args):
    try: return max(*args)
    except ValueError: return 0.

def _day_frac_2_str(frac_hour):
    hour = floor(frac_hour)
    mins = floor(frac_hour - hour)
    return '{}:{}'.format(str(hour).zfill(2), str(mins).zfill(2))

def _tick_locs(day_start, day_end):
    frac_times = np.arange(day_start, day_end, 1.)
    ats = (frac_times - day_start) / (day_end - day_start)
    lbs = [_day_frac_2_str(at) for at in frac_times]
    return ats.tolist(), lbs

def _temporal_shedding_patches(
        boxes,
        day_start,
        day_end,
        interval_getter,
        return_patches = True,
        patch_data = None):

    return_patches = return_patches or patch_data is not None

    if return_patches:
        def make_patch(interval, xy, mx_height, scale_width=1., scale_height=1.):
            return mpatches.Rectangle(
                (xy[0] + scale_width * interval.start, xy[1]),
                interval.length * scale_width,
                interval.shedding / max_cum_height)
    else:
        def make_patch(interval, xy, mx_height, scale_width=1., scale_height=1.):
            return interval, xy, mx_height, scale_width, scale_height

    if patch_data is None:
        scaler           = 1. / (day_end - day_start)
        box_heights      = [_safe_max(i.shedding for i in interval_getter(box)) for box in boxes]
        cum_box_heights  = np.cumsum([0,] + box_heights)
        max_cum_height   = cum_box_heights[-1]
        box_base_heights = cum_box_heights[:-1] / max_cum_height

        return [make_patch(i, (-day_start * scaler, bbh), max_cum_height, scaler) \
            for bbh, box in zip(box_base_heights, boxes) \
                for i in interval_getter(box)], box_base_heights

    else:
        patches, box_base_heights = patch_data
        return [make_patch(*patch) for patch in patches], box_base_heights



def extract_plot_data(
        workplace = None,
        boxes = None,
        ventilation_system = None,
        day_start = None,
        day_end = None,
        human_box_types = None,
        extract_boxes = True,
        extract_concentrations = True,
        extract_expected_infections = False,
        pathogen_dieoff_rate = DEFAULT_PATHOGEN_DIEOFF_RATE):

    assert extract_boxes or extract_concentrations or extract_expected_infections

    if workplace is None:
        assert boxes is not None
        assert ventilation_system is not None
        assert day_start is not None
        assert day_end is not None
    else:
        assert boxes is None
        assert ventilation_system is None
        assert day_start is None
        assert day_end is None
        day_start = workplace.day_start
        day_end   = workplace.day_end
        boxes = workplace.boxes
        ventilation_system = workplace.ventilation_system

    if human_box_types is not None:
        boxes = [b for b in boxes if b.use in human_box_types]

    if extract_boxes:
        # patches for people
        kept_index_map = {b.box_index:i for i, b in enumerate(boxes)}
        schedules = [s for s in workplace.infected_schedules.values()]
        color_array = np.asarray([kept_index_map[i.box.box_index] for s in schedules for i in s])
        person_patches = *_temporal_shedding_patches(
            schedules,
            day_start, day_end,
            lambda x: x), color_array

        # patches for boxes
        color_array = np.asarray(reduce(
            lambda c, b: c + ([b[0]] * len(b[1].infected_intervals)),
            enumerate(boxes),
            []))
        box_patches = *_temporal_shedding_patches(
            boxes,
            day_start, day_end,
            lambda box: box.infected_intervals), color_array

    else:
        person_patches = None
        box_patches = None

    if extract_concentrations:
        kept_index_map = {b.box_index:i for i, b in enumerate(boxes)}
        # concentrations and times
        times = np.arange(day_start, day_end, ONE_MINUTE)
        scaled_times = (times - day_start) / (day_end - day_start)
        concentrations = [[b.concentration_at_time(t, pathogen_dieoff_rate) for t in times]
            for b in boxes]
        concentration_data = times, scaled_times, concentrations

    else:
        concentration_data = None

    # risk in boxes
    if extract_expected_infections:
        box_risks = [box.total_exposure_risk for box in boxes]
    else:
        box_risks = None

    return {
        'day_start': day_start,
        'day_end': day_end,
        'n_boxes': len(boxes),
        'box_patches': box_patches,
        'person_patches': person_patches,
        'concentration_data': concentration_data,
        'box_risks': box_risks,
        'box_names': [b.name for b in boxes],
        'box_uses': [b.use for b in boxes]}



def plot_concentrations(
        ax,
        ventilation_system = None,
        boxes = None,
        day_start = None,
        day_end = None,
        pathogen_dieoff_rate = DEFAULT_PATHOGEN_DIEOFF_RATE,
        title = None,
        x_axs_title = None,
        ylim = None,
        cmap = None,
        alpha = .6,
        tick_locs = None,
        tick_labs = None,
        include_legend = False,
        data = None):

    if data is None:
        assert ventilation_system is not None
        assert boxes is not None
        data = extract_plot_data(
            ventilation_system = ventilation_system,
            boxes = boxes,
            day_start = day_start,
            day_end = day_end,
            extract_boxes = False,
            pathogen_dieoff_rate = pathogen_dieoff_rate)

    else:
        assert ventilation_system is None
        assert boxes is None

    n_boxes = data['n_boxes']

    if day_start is None:
        day_start = data['day_start']

    if day_end is None:
        day_end = data['day_end']

    if cmap is None:
        cmap = mpl.cm.get_cmap('jet', n_boxes)

    if tick_locs is None:
        assert tick_labs is None
        tick_locs, tick_labs = _tick_locs(day_start, day_end)
    else:
        assert tick_labs is not None

    times, scaled_times, concentrations = data['concentration_data']

    for i, cs in enumerate(concentrations):
        ax.plot(scaled_times, cs, color=cmap(i), alpha=alpha)
    ax.axes.get_xaxis().set_ticks(tick_locs)
    ax.axes.get_xaxis().set_ticklabels(tick_labs)
    ax.set(ylabel='quanta/m${}^3$')
    if ylim is not None:
        ax.set_ylim(ylim)

    if x_axs_title is not None:
        ax.set(xlabel=x_axs_title)

    if title is not None:
        ax.title.set_text(title)

    if include_legend:
        legend_patches = [mpatches.Patch(
            color=cmap(i),
            label=name,
            alpha=alpha) for i, name in enumerate(data['box_names'])]

        plt.legend(
            handles=legend_patches,
            ncol=n_boxes,
            bbox_to_anchor=(.5, -.15),
            loc='upper center',
            frameon=False,
            fancybox=False)



def shedding_plot(
        workplace,
        file_path=None,
        human_box_types = None,
        pathogen_dieoff_rate=DEFAULT_PATHOGEN_DIEOFF_RATE,
        show_boxes=True,
        show_concentrations=True,
        show_expected_infections=False,
        show_expected_infections_by_box_type=False,
        include_legend=True,
        show_plot=True,
        panel_height=4.):

    alpha = .6
    n_subplots = 0
    conc_plot_index = None
    inf_plot_index = None
    legend_space = -0.08

    if show_boxes:
        n_subplots += 2

    if show_concentrations:
        conc_plot_index = n_subplots
        n_subplots += 1

    if show_expected_infections:
        inf_plot_index = n_subplots
        n_subplots += 1
        legend_space = -0.21

    if n_subplots == 0:
        return None

    plot_data = extract_plot_data(
        workplace = workplace,
        human_box_types = human_box_types,
        extract_boxes = show_boxes,
        extract_concentrations = show_concentrations,
        extract_expected_infections = show_expected_infections,
        pathogen_dieoff_rate = pathogen_dieoff_rate)

    n_boxes = plot_data['n_boxes']
    day_start = plot_data['day_start']
    day_end = plot_data['day_end']

    my_cmap = mpl.cm.get_cmap('jet', n_boxes)
    locs, labs = _tick_locs(day_start, day_end)

    fig, axs = plt.subplots(n_subplots, figsize=(10, n_subplots * panel_height))

    if show_boxes:
        # The schedules of infected people
        patches, line_ys, color_array = plot_data['person_patches']
        p = PatchCollection([p for p in patches], cmap=my_cmap, alpha=alpha)
        p.set_array(color_array)
        p.set_clim(0., n_boxes)
        ax = axs[0]
        ax.title.set_text('Shedding From Individuals')
        #ax.set(ylabel='Shedding')
        ax.add_collection(p)
        ax.hlines(
            y=line_ys,
            xmin=0., xmax=1.,
            linewidth=1, linestyle='dashed',
            color='grey')
        plt.tight_layout()
        ax.axes.get_xaxis().set_ticks(locs)
        ax.axes.get_xaxis().set_ticklabels(labs)
        ax.axes.get_yaxis().set_ticks([])

        # The boxes
        patches, line_ys, color_array = plot_data['box_patches']
        p = PatchCollection(patches, cmap=my_cmap, alpha=alpha)
        p.set_array(color_array)
        p.set_clim(0., n_boxes)
        ax = axs[1]
        ax.title.set_text('Shedding In Boxes')
        #ax.set(ylabel='Shedding')
        ax.add_collection(p)
        ax.hlines(
            y=line_ys,
            xmin=0., xmax=1.,
            linewidth=1, linestyle='dashed',
            color='grey')
        plt.tight_layout()
        ax.axes.get_xaxis().set_ticks(locs)
        ax.axes.get_xaxis().set_ticklabels(labs)
        ax.axes.get_yaxis().set_ticks([])

    if conc_plot_index is not None:
        if n_subplots > 1:
            ax = axs[conc_plot_index]
        else:
            ax = axs

        # concentrations in boxes
        plot_concentrations(
            ax = ax,
            pathogen_dieoff_rate = pathogen_dieoff_rate,
            title = 'Concentration In Boxes',
            x_axs_title = 'Time of Day',
            cmap = my_cmap,
            alpha = alpha,
            tick_locs = locs,
            tick_labs = labs,
            include_legend = False,
            data = plot_data)

    if inf_plot_index is not None:
        if n_subplots > 1:
            ax = axs[inf_plot_index]
        else:
            ax = axs

        if show_expected_infections_by_box_type:
            counts = defaultdict(list)
            for use, risk in zip(plot_data['box_uses'], plot_data['box_risks']):
                counts[use].append(risk)
            counts_tuples = [t for t in counts.items()]
            subber = re.compile(r'\.(.*)')
            names = [subber.search(str(t[0])).group(1) for t in counts_tuples]
            bar_heights = [np.mean(t[1]) for t in counts_tuples]
            title = 'Average Number of Infections in Box Type from Exposure in That Box Type'
            label_rotation = 0
            colors = 'grey'

        else:
            names = plot_data['box_names']
            bar_heights = plot_data['box_risks']
            title = 'Average Number of Infections in Box Type from Exposure in That Box Type'
            title = 'Expected Infections From Exposure in Box'
            label_rotation = 60
            colors = [my_cmap(i) for i in range(len(names))]

        n_bars = len(names)

        ax.title.set_text(title)
        ax.set(ylabel='Infections')
        ax.bar(
            x = [i for i in range(1, n_bars + 1)],
            height = bar_heights,
            tick_label = names,
            color = colors,
            alpha = alpha)
        ax.tick_params(axis='x', rotation=label_rotation)

    # legend
    if include_legend:
        legend_patches = [mpatches.Patch(
            color=my_cmap(i),
            label=name,
            alpha=alpha) for i, name in enumerate(plot_data['box_names'])]

        plt.legend(
            handles=legend_patches,
            ncol=n_boxes,
            bbox_to_anchor=(.5, legend_space),
            loc='upper center',
            frameon=False,
            fancybox=False)

    fig.tight_layout()
    if file_path is not None:
        plt.savefig(file_path)#, bbox_inches='tight')

    if show_plot:
        plt.show()
