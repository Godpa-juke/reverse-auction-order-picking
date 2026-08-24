import numpy as np


def Make_Maze(env, mode):
    """Generate occupancy maps for different planning modes.

    Relies on the environment to expose layer indices via its config.
    """

    cfg = getattr(env, "config", None)
    layer_agents = getattr(env, "layer_agents", None)
    layer_shelfs = getattr(env, "layer_shelfs", None)
    layer_spots = getattr(env, "layer_spots", None)

    if cfg is not None:
        layer_agents = layer_agents if layer_agents is not None else cfg.layer_agents
        layer_shelfs = layer_shelfs if layer_shelfs is not None else cfg.layer_shelfs
        layer_spots = layer_spots if layer_spots is not None else cfg.layer_spots
    else:
        raise AttributeError("Environment is missing SimulationConfig. Cannot build maze.")

    walls = getattr(env, "walls", None)
    enforce = getattr(cfg, "wall_enforce_level", 0) == 2

    arr = np.zeros(env.grid_size)
    if mode == 0:
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                if env.grid[layer_agents, i, j] > 0 and env.grid[layer_agents, i, j] <= env.n_max_humans:
                    arr[i][j] = 1

                if env.grid[layer_spots, i, j] in [3, 5]:
                    arr[i][j] = 1

                if enforce and walls is not None and walls[i][j] == 2:
                    arr[i][j] = 1

    elif mode == 1:  # For Human
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                if env.grid[layer_shelfs, i, j] > 0 and env.grid[layer_agents, i, j] <= env.n_max_humans:
                    arr[i][j] = 1

                if env.grid[layer_spots, i, j] in [1, 3, 4, 5]:
                    arr[i][j] = 1

                if enforce and walls is not None and walls[i][j] == 2:
                    arr[i][j] = 1

    elif mode == 2:  # For Robot
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                if env.grid[layer_agents, i, j] > env.n_max_humans:
                    arr[i][j] = 1

                if env.grid[layer_shelfs, i, j] > 0:
                    arr[i][j] = 1

                if env.grid[layer_spots, i, j] in [3, 5, 8, 9]:
                    arr[i][j] = 1

                if enforce and walls is not None and walls[i][j] >= 1:
                    arr[i][j] = 1

    elif mode == 3:  # For Robot
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                if env.grid[layer_agents, i, j] > env.n_max_humans:
                    arr[i][j] = 1

                if env.grid[layer_shelfs, i, j] > 0:
                    arr[i][j] = 1

                if env.grid[layer_spots, i, j] in [3, 5]:
                    arr[i][j] = 1

                if enforce and walls is not None and walls[i][j] >= 1:
                    arr[i][j] = 1

    else:
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                if env.grid[layer_shelfs, i, j] > 0:
                    arr[i][j] = 1

                if env.grid[layer_spots, i, j] in [3, 5]:
                    arr[i][j] = 1

                if enforce and walls is not None and walls[i][j] == 2:
                    arr[i][j] = 1
    return arr
