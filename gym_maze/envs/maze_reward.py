import numpy as np

def calculate_reward_original(self):
    if np.array_equal(self.maze_view.robot, self.maze_view.goal):
        reward = 1
        done = True
    else:
        reward = -0.1/(self.maze_size[0]*self.maze_size[1])
        done = False
    return reward, done

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