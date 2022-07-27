[Return to parent](../README.md)

# Questacon: A Public Science and Technology Museum

Questacon is a science and technology museum located in Canberra, ACT, Australia.




## Contents

- ***boxes.csv***: A CSV describing each room in Questacon and the volume of air exchanged
  directly between it and the external environment.

  The columns in the file are:

    - first column: An index over the rows. This is ignored.

    - ***room***: The name of one of the room.

    - ***use***: The type of the room.

    - ***volume***: The volume of the *room* in cubic meters.

    - ***ventilation_out***: The volume of the air vented from the room directly to the outside
      environment in cubic meters per hour.

    - ***ventilation_in***: The volume of the air pumped directly into the room from the outside
      environment in cubic meters per hour.

- ***flows.csv***: A CSV describing the air movements between the various rooms.

  The columns in the file are:

    - ***from***: The room the air flows from. This must correspond to one of the names in the
      column *room* of *boxes.csv*.

    - ***to***: The room the air flows to. This must correspond to one of the names in the
      column *room* of *boxes.csv*.

    - ***volume***: The volume of air that flows between the rooms.

- ***simulation.ipynb***: Jupyter notebook for running various scenarios.

- ***utils.py***: Utilities for creating instances of the office and specifying
  the behaviour of agents.
