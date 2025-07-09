import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

maze = [
    "#########",
    "#S  #   #",
    "# ## ## #",
    "#    #  #",
    "####   G#",
    "#########",
]

height = len(maze)
width = len(maze[0])
start = (1, 1)
goal = (7, 4)
move_map = [(0, -1), (1, 0), (0, 1), (-1, 0)]
steps = 40

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_ind():
    return creator.Individual(random.choices(range(4), k=steps))

def eval_maze(ind):
    x, y = start
    for m in ind:
        dx, dy = move_map[m]
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] != '#':
            x, y = nx, ny
    dist = abs(goal[0] - x) + abs(goal[1] - y)
    return dist,

def decode(ind):
    x, y = start
    path = [start]
    for m in ind:
        dx, dy = move_map[m]
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] != '#':
            x, y = nx, ny
            path.append((x, y))
    return path


toolbox = base.Toolbox()
toolbox.register("individual", create_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_maze)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def plot_path(path):
    grid = np.array([list(row) for row in maze])
    xs, ys = zip(*path)
    grid[ys, xs] = '.'
    grid[start[1]][start[0]] = 'S'
    grid[goal[1]][goal[0]] = 'G'
    plt.imshow(grid != '#', cmap="Greys")
    plt.plot(xs, ys, 'r-')
    plt.show()


def main():
    pop = toolbox.population(100)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 80, halloffame=hof, verbose=False)
    best = hof[0]
    print("distance", eval_maze(best)[0])
    plot_path(decode(best))


if __name__ == "__main__":
    main()
