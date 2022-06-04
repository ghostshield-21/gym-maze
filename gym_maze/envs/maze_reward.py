import numpy as np
import math

# PPO: works ok early, but overtime, due to high possitive reward, low negative reward, model tend to increase steps 
def calculate_reward_xy_latest(self):

    if self.maze_view.maze.is_pick( tuple(self.maze_view.robot) ):
        self.maze_view.maze.load_pick( tuple(self.maze_view.robot) )
        reward = 1 / self.maze_view.maze.num_picks
        done = False

    elif np.array_equal(self.maze_view.robot, self.maze_view.goal):
        loaded_total = len(self.maze_view.maze.get_loaded_picks())
        reward = (1/ self.maze_view.maze.num_picks) * ( loaded_total**loaded_total )
        print("**Complete**, Picked up packages:",len(self.maze_view.maze.get_loaded_picks()) )
        print("Last reward:", reward)
        
        done = True

    else:
        reward = -0.5/(self.maze_size[0]*2)
        done = False
    
    return reward, done



def calculate_reward_original(self):
    if np.array_equal(self.maze_view.robot, self.maze_view.goal):
        reward = 1
        done = True
    else:
        reward = -0.1/(self.maze_size[0]*self.maze_size[1])
        done = False
    return reward, done


# PPO: works ok, but does not always pickup all the packages, avg steps 110
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


# PPO: works ok early, but overtime, due to high possitive reward, low negative reward, model tend to increase steps 
def calculate_reward_xy_2(self):

    if self.maze_view.maze.is_pick( tuple(self.maze_view.robot) ):
        self.maze_view.maze.load_pick( tuple(self.maze_view.robot) )
        reward = 1 / self.maze_view.maze.num_picks
        done = False

    elif np.array_equal(self.maze_view.robot, self.maze_view.goal):
        reward = len(self.maze_view.maze.get_loaded_picks())
        done = True
        print("******************compeleted******************")
        print("Last reward", reward)

    else:
        reward = -0.1/(self.maze_size[0]*self.maze_size[1])
        done = False
    
    return reward, done
