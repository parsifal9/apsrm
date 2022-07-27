[Return to parent](../README.md)

# Eveleigh

A more complicated office based on the top floor of an office building located
in Eveleigh, NSW, Australia.




## Contents

- ***boxes.csv***: A CSV describing the connections between the various rooms
  in the office. Each row describes the movement of air between two rooms. Only
  cases where the airflow is non-zero in at least one direction are included.

  The columns in the file are:

    - first column: An index over the rows. This is ignored.

    - ***room1***: The name of one of the rooms.

    - ***room1.use***: What *room1* is used for.

    - ***room1.area***: The floor area of *room1*. This is used for calculating
      the area of *room1* (the ceiling height is assumed to be three meters in
      all rooms).

    - ***room1.occupancy***: The maximum number of people who can use *room1*.
      This is only used for meeting rooms (i.e. all other values are ignored).

    - ***room2***: The name of the other room.

    - ***room2.use***: What *room2* is used for.

    - ***room2.area***: The floor area of *room2*. This is used for calculating
      the area of *room2* (the ceiling height is assumed to be three meters in
      all rooms).

    - ***room2.occupancy***: The maximum number of people who can use *room2*.
      This is only used for meeting rooms (i.e. all other values are ignored).

    - ***ACP.1.2***: The volume of air (in cubic meters per hour) flowing from
      *room1* to *room2*

    - ***ACP.2.1***: The volume of air (in cubic meters per hour) flowing from
      *room1* to *room2*

    - ***code***: Ignored.

    - ***air.volume.moved***: Ignored.

- ***simulation.ipynb***: Jupyter notebook for running various scenarios.

- ***utils.py***: Utilities for creating instances of the office and specifying
  the behaviour of agents.
