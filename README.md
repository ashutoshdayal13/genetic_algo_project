# Genetic Algorithm 

This project shows how to use genetic algorithms with the DEAP library.
It includes a Vehicle Routing Problem solver and a small maze path finder.

## Setup

Install requirements with:
```bash
pip install -r requirements.txt
```

## Vehicle Routing Problem

Run `python vrp_ga.py` to search for a near optimal set of routes.
The script prints the best distance and draws the final tour using Matplotlib.

## Maze Solver

Run `python maze_ga.py` to evolve a path through a grid maze.
The best found path is displayed as a plot.
