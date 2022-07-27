[return to parent](../README.md)

# Data



## Contents
- ***Emmission.csv***: Same data as *Emission.xlsx*, but as CSV.

    **Units**:

    - *activity*: What the person is doing.
    - *virons*: Virons per hour.
    - *breathing_rate*: cubic meters per hour.

- ***Emmission.xlsx***: Emissions data prepared by Jess Liebig. The column *Virons* is
  specified in this is copied. I think that dividing by 64,000 converts this to quanta.

- ***load-emissions.py***: Load the data in *Emission.csv*, convert it to JSON and save it in
  *../apsrm/_emissions.json*.

- ***pcr-beta-bernoulli.R***: R script to estimate the parameters of beta distributions that
  describe the sensitivity of a PCR test (results written to *../apsrm/_prc-beta-params.csv*).

- ***prc-sensitivity.rds***: The raw data used by *pcr-beta-bernoulli.R*.
