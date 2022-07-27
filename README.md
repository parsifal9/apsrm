# apsrm

An agent based model originally developed for assessing potential interventions to mitigate the risk
of COVID-19 transmission **via aerosols** in workplaces, **assuming all rooms within the workplace
are well mixed**. Risk of transmission from other mechanisms (e.g. fomites or breathing exhaled air
directly) are not considered.

The name is an acronym for ***Airborne Pathogen Spread Risk Model***.

The framework has been designed to separate the specification of the behaviour of individuals and
the characteristics of different workplaces. The core of the framework simply provides a set of
algorithms that execute the following steps (in order) in each day:

1. track the movement of individuals through rooms which comprise the workplace;

1. track the quanta emissions of each infected individual in each room, through time;

1. calculate the resulting concentration in each room over the course of a day;

1. track the exposure of the susceptible individuals to over the course of the day;

1. estimate the aggregate infection risk to each susceptible individual; and

1. randomly infects susceptible individuals based on their infection risk.




## Getting Started

On a proper operating system (in this case, one with [make](https://www.gnu.org/software/make/)
installed), after cloning this repository, you should be able to simply run

```bash
make initial-setup
```

This will create a virtual environment at *./venv* with the package installed for development (which
means you can work on it and the changes will be reflected in the installation). It will also
install everything you need for building the package and running the [Jupyter](https://jupyter.org/)
notebooks, which can be found in the subfolders under *./workplaces*.

In order to run analyses and alike you will need to activate the virtual environment with

```bash
. venv/bin/activate
```

The easiest place to get started is in the office, with something like

```bash
. venv/bin/activate
cd workplaces/office
jupyter notebook
```

which should open a browser in that folder. The notebook *simulation* provides examples on how one
might run simulations. Please [read the docs](https://jupyter-notebook.readthedocs.io/en/stable/) to
learn about working with Jupyter.




## Dependencies

- Python 3 (we started at 3.6).

- [virtualenv](https://virtualenv.pypa.io/en/latest/) for making environments, though you can
  probably just use the *venv* module as per the advice at that link. This is not required, but life
is much easier with a virtual environment.

- To generate the data used by the PCR test (which you can do with `make data`), you will need
  [Rscript](https://www.rdocumentation.org/packages/utils/versions/3.6.2/topics/Rscript) (installed
  as part of the standard R installation) with the packages
  [fitdistrplus](https://cran.r-project.org/package=fitdistrplus) and
  [rjson](https://CRAN.R-project.org/package=rjson). The processed data is versioned, so you would
  only need to do this if you change that data.




## Contents



### Folders

- ***[data](./data/README.md)***: Static data and scripts for working with it.

- ***doc***: Sphinx documentation and configuration. You can generate the documentation with `make
  sphinx-doc`. You can once built, you can find the documentation at
  [./doc/build/index.html](./doc/build/index.html).

- ***apsrm***: Code for the Python package. Run `make initial-setup` or `make venv` to install it
  for development.

- ***outputs***: Default output directory for results generated using the Jupyter notebooks under
  *workplaces*.

- ***tests***: Tests. After installation (with `.[dev]`), run `pytest` to run the tests.

- ***[workplaces](./workplaces/README.md)***: Models for different workplaces. Each workplace is in
  a separate folder. The analyses themselves are in [Jupyter](https://jupyter.org) notebooks. The
  file *workplaces/run-all.py* will run the notebooks required to produce outputs for the original
  paper, but they may, of course, be run interactively.



### Files

- ***Dockerfile***: Docker build file.

- ***Makefile***: Make file for common tasks. This is good place to see how to do things, even if
  you haven't got a reasonable operating system that doesn't support make.

- ***MANIFEST.in***: Tells *setup.py* what files should be installed with the package.

- ***setup.py***: The setup script for the package.

- ***version***: Contains a version number for the Python package.

- ***./apsrm/emissions.json***: Data on breathing rates and pathogen emissions rates for
  different activities by variant. Produced by the script *./data/load-emissions.py*.

- ***./apsrm/prcbetaparams.json***: The estimates of the Beta parameters produced by
  *./data/pcr-beta-bernoulli.R*.

- ***./apsrm/vaccineefficacy.json***: Daily vaccine efficacy (proportional reduction in the
  probability of becoming infected from an exposure) for the first 731 days after vaccination.
  Produced by the script *./data/vaccine-efficacy.R*.

- ***run-all.py***: A script that runs the analyses presented in our JRSI paper... remember to
  activate your venv before running this!

The implementation is mostly in private modules under *./apsrm* and the public API is manged in
*./apsrm/__init__.py*. Please follow your nose from there.




## Misc Stuff



### Docker

Provided for conveniences, but might be useful if you don't have a proper operating system but can
install [Docker](https://www.docker.com/).  Build with

```bash
docker build -t diser .
```

To then run a Jupyter notebook (available at port 8888 on the host) from a Docker container

```bash
docker run -it --rm \
    -p 8888:8888 \
    -v `pwd`:/nbs \
     diser \
         jupyter notebook --allow-root --ip=0.0.0.0 --no-browser --notebook-dir=/nbs
```



### Developer Documentation

```make sphinx-doc```

Generated documenation can be found at *./doc/build/index.html*.



### Optional Requirements

#### dev

Optional requirements for development. At the moment this is stuff for building the documentation
and running tests. You will need this for make commands *sphinx-doc* and *initial-setup*.


#### notebooks

Optional requirements for running the [Jupyter](https://jupyter.org/) notebooks found under
*workplaces*.
