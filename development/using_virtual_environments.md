# Using virtual environments
The benefit of using virtual enviornments is being able to maintain the requirements and imports for a given project. You can easily export the requirements that were used for a project to allow it to run and then make those requirements available to other programmers by using `pip freeze` later on. 

You can create a virtual environment using `python3 -m venv venv`; this creates an environment called `venv`. You can then activate this using `source venv/bin/activate`. 