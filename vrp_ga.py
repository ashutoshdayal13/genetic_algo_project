import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# demand, x, y
customers = [
    (2, 2, 7),
    (3, 5, 8),
    (5, 1, 3),
    (2, 6, 2),
    (4, 8, 4),
    (1, 7, 9),
    (2, 3, 2),
    (3, 9, 6),
    (4, 4, 7),
    (2, 2, 4),
]

vehicle_cap = 15
num_vehicles = 3
depot = (5, 5)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def create_ind():
    return creator.Individual(random.sample(range(len(customers)), len(customers)))


def decode(ind):
    routes = []
    cap = 0
    r = []
    for idx in ind:
        d, x, y = customers[idx]
        if cap + d > vehicle_cap:
            routes.append(r)
            r = []
            cap = 0
        r.append(idx)
        cap += d
    if r:
        routes.append(r)
    return routes


def dist(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def eval_vrp(ind):
    routes = decode(ind)
    if len(routes) > num_vehicles:
        return 1e6,
    total = 0.0
    for r in routes:
        last = depot
        for idx in r:
            _, x, y = customers[idx]
            total += dist(last, (x, y))
            last = (x, y)
        total += dist(last, depot)
    return total,


toolbox = base.Toolbox()
toolbox.register("individual", create_ind)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_vrp)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def plot_routes(routes):
    colors = "rgbcmyk"
    dx, dy = depot
    plt.scatter(dx, dy, c="k", marker="s", label="depot")
    for i, (d, x, y) in enumerate(customers):
        plt.scatter(x, y, c="gray")
        plt.text(x + 0.1, y + 0.1, str(i))
    for i, r in enumerate(routes):
        c = colors[i % len(colors)]
        x, y = dx, dy
        for idx in r:
            _, cx, cy = customers[idx]
            plt.plot([x, cx], [y, cy], c)
            x, y = cx, cy
        plt.plot([x, dx], [y, dy], c, label=f"veh {i + 1}")
    plt.legend()
    plt.axis("equal")
    plt.show()


def main():
    pop = toolbox.population(80)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, 0.7, 0.3, 150, halloffame=hof, verbose=False)
    best = hof[0]
    print("best distance", eval_vrp(best)[0])
    plot_routes(decode(best))


if __name__ == "__main__":
    main()
