from tkinter import E
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_maze.envs.maze_view_2d import MazeView2D
import gym_maze.envs.maze_reward as maze_reward

"""
Warehouse Version v0.1
   changed observation state and state of the warehouse to included
        payloads
        robot
        goal

    added a simple reward based on the the new state structure 

"""


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True):
        print('version v0.1 warehouse')
        self.viewer = None
        self.done=False
        self.enable_render = enable_render
        
        if mode=="pick":
            num_picks= 4
        else:
            num_picks =0

        if maze_file:
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640), 
                                        enable_render=enable_render, num_picks=num_picks)
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size)/3))
            else:
                has_loops = False
                num_portals = 0

            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros( self.maze_size, dtype=int)
        high =  np.full( self.maze_size, 3) 
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        # initial condition
        self.state = np.zeros( self.maze_size, dtype=int)
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = int(action)
        if isinstance(action, int):
            self.maze_view.move_robot(self.ACTION[action])
        else:
            print(action)
            self.maze_view.move_robot(action)

        reward, done = self.calculate_reward_xy_1()

        self.calculate_state()
        
        info = {}

        return self.state, reward, done, info

    def calculate_reward_xy_1(self):

        if self.maze_view.maze.is_pick( tuple(self.maze_view.robot) ):
            self.maze_view.maze.load_pick( tuple(self.maze_view.robot) )
            reward = 1 / self.maze_view.maze.num_picks
            done = False

        elif np.array_equal(self.maze_view.robot, self.maze_view.goal):
            reward = 1
            done = True
            print("******************compeleted******************")
            print("loaded packages: ", self.maze_view.maze.get_loaded_picks())

        else:
            reward = -0.1/(self.maze_size[0]*self.maze_size[1])
            done = False

        return reward, done


    def calculate_state(self):

        self.state = np.zeros( self.maze_size, dtype=int)
        self.state[self.maze_view.robot[0], self.maze_view.robot[1]] = 1
        self.state[self.maze_view.goal[0], self.maze_view.goal[1]] = 3

        for pick_corrd in self.maze_view.maze.get_picks_all():
            self.state[pick_corrd[0], pick_corrd[1]] = 2
        
    def reset(self):
        self.maze_view.reset_robot()
        self.calculate_state()
        self.steps_beyond_done = None
        self.done = False
        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)

class MazeWarehouse(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeWarehouse, self).__init__(maze_file="maze_warehouse.npy", enable_render=enable_render)

class MazeWarehousePicks(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeWarehousePicks, self).__init__(maze_file="maze_warehouse.npy", mode="pick", enable_render=enable_render)

class MazeEnvSample5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", enable_render=enable_render)


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5), enable_render=enable_render)


class MazeEnvSample10x10(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy", enable_render=enable_render)


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10), enable_render=enable_render)


class MazeEnvSample3x3(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy", enable_render=enable_render)


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3), enable_render=enable_render)


class MazeEnvSample100x100(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy", enable_render=enable_render)


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100), enable_render=enable_render)


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus", enable_render=enable_render)


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus", enable_render=enable_render)


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self, enable_render=True):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus", enable_render=enable_render)
