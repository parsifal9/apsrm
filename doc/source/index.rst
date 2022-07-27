.. apsrm documentation master file, created by
   sphinx-quickstart on Wed Aug 11 07:42:58 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to apsrm's documentation!
====================================

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Package
=======

.. toctree::
   :maxdepth: 2

   apsrm

Data
====

.. toctree::
   :maxdepth: 2

   emissions
   pcr-test-data
   vaccine-efficacy

Overview
========

An agent based modelling framework originally developed for assessing potential interventions to
mitigate the risk of COVID-19 transmission risk in workplaces.

The framework has been designed to separate the specification of the behaviour of individuals and
the characteristics of different workplaces. The core of the framework simply provides a set of
algorithms that execute the following steps (in order) in each day:

#. track the movement of individuals through rooms which comprise the workplace;

#. track the quanta emissions of each infected individual in each room, through time;

#. calculate the resulting concentration in each room over the course of a day;

#. track the exposure of the susceptible individuals to over the course of the day;

#. estimate the aggregate infection risk to each susceptible individual; and

#. randomly infects susceptible individuals based on their infection risk.

To execute this, create a :py:class:`apsrm.Workplace`, and call the method
:py:meth:`apsrm.Workplace.run_period` on it, which runs a single period and can be thought of as
the main entrypoint for models using the framework.

:py:meth:`apsrm.Workplace.run_period` first calls the method
:py:meth:`apsrm.GatheringGenerator.create_gatherings` on any
:py:class:`apsrm.GatheringGenerator` instances which have been registered on the workplace (using
the method :py:meth:`apsrm.Workplace.add_generator`). This process generates a list of
:py:class:`apsrm.Gathering`\ s, which are later used as the basis of to :term:`schedule`\ s for
the agents attending the gatherings.

:py:meth:`apsrm.Person.generate_schedule` is then called on every agent either working in or
visiting the workplace, generating :term:`schedule`\ s for each.

The :term:`schedule`\ s for the infectious agents are then used to calculate the total shedding in
each of the rooms in the workplace. Then the concentration of airborne virus in each room over the
duration of the period is calculated by calling the method
:py:meth:`apsrm.ventilation.VentilationSystem.calculate_concentrations`.

The :term:`schedule`\ s for the susceptible agents are used to calculate their exposure over the
duration of the period, and hence the probability that they become infected, and finally, they are
randomly infected with this probability.



Glossary
========

This glossary is tuned to how we use the terms within CVTS, which is often different on how they are
used more generally.

.. glossary::

    appointment
        A place an agent has to be through some period of time. Represented
        by the class :py:class:`apsrm.interval.TimeInterval`.

    schedule
        A list of :term:`appointment`\ s.
