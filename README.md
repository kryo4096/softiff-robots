# softiff-robots
soft differentiable robots


How to run

----------

* [Download Julia and make sure to add it to your PATH](https://julialang.org/downloads/)
* Download the repository, open a terminal and navigate to the root folder of the project
* run the command `julia` to start an interactive julia session
* To install the Packages needed, open a terminal in the project folder and type `]` to enter Pkg REPL Mode
* Enter `activate`, then `instantiate` to download all the packages specified in the environment (This could take a while)
* Exit the Package mode by hitting Backspace and now you are ready to run the simulation.
* If you simply want to run the simulation, run the command `include("src/main.jl");run_simulation()` which will start the simulation with the pre-generated robot
