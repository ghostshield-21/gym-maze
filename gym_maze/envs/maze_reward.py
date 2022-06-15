import numpy as np
import math

# reward function v5
def calculate_reward_xy_latest(self):
    done = False

    # reward for picking up a package
    # based on linear relations with how many packaged been picked up 
    if self.maze_view.maze.is_pick( tuple(self.maze_view.robot) ):
        total_loaded = len(self.maze_view.maze.get_loaded_picks())
        self.maze_view.maze.load_pick( tuple(self.maze_view.robot) )
        reward = 50 + 50*total_loaded
        print("**Loading**, Picked up packages:", len(self.maze_view.maze.get_loaded_picks()), "  step reward: ", reward)

    # minimal possitive reward for arriving terminal location
    elif np.array_equal(self.maze_view.robot, self.maze_view.goal):
        reward = 1
        done = True
    
    # negative reward for taking a step
    else:
        reward = -1

    
    return reward, done

# reward function v4 
def calculate_reward_xy_4(self):
    done = False
    loaded_total = len(self.maze_view.maze.get_loaded_picks())

    # fixed reward for picking up an package
    if self.maze_view.maze.is_pick( tuple(self.maze_view.robot) ):
        self.maze_view.maze.load_pick( tuple(self.maze_view.robot) )
        reward = 20

    # fixed reward for arriving terminal location
    elif np.array_equal(self.maze_view.robot, self.maze_view.goal):

        # fixed reward based on numbers of package been loaded.   
        if np.array_equal(self.maze_view.robot, self.maze_view.goal):
            if loaded_total==self.maze_view.maze.num_picks:
                reward =  200
                print("**Complete**, Picked up packages:", len(self.maze_view.maze.get_loaded_picks()), "  Last step reward: ", reward)
                done = True
            elif loaded_total>=(self.maze_view.maze.num_picks-1):
                reward =  80
                print("**Complete**, Picked up packages:", len(self.maze_view.maze.get_loaded_picks()), "  Last step reward: ", reward)
                done = True
            elif loaded_total>=(self.maze_view.maze.num_picks-2):
                reward =  50
                print("**Complete**, Picked up packages:", len(self.maze_view.maze.get_loaded_picks()), "  Last step reward: ", reward)
                done = True
            
            # negative reward if there is not package been loading when arriving terminal location
            else:
                done = True
                reward =  -200

    # fixed negative reward per step
    else:
        reward = -0.5
    
    return reward, done


# reward function v3
def calculate_reward_xy_3(self):

    # fixed reward for picking up an package, while less than arriving at terminal location
    if self.maze_view.maze.is_pick( tuple(self.maze_view.robot) ):
        self.maze_view.maze.load_pick( tuple(self.maze_view.robot) )
        reward = 1 / self.maze_view.maze.num_picks
        done = False

    # reward for arriving terminal location
    # based on Exponential function of how many packaged is been loaded.  
    elif np.array_equal(self.maze_view.robot, self.maze_view.goal):
        loaded_total = len(self.maze_view.maze.get_loaded_picks())
        reward = (1/ self.maze_view.maze.num_picks) * ( loaded_total**loaded_total )
        print("**Complete**, Picked up packages:",len(self.maze_view.maze.get_loaded_picks()), "  Last step reward: ", reward)
        done = True

    # fixed negative reward per step
    else:
        reward = -0.5/(self.maze_size[0]*2)
        done = False
    
    return reward, done


# original reward function, not been used for warehouse
def calculate_reward_original(self):
    if np.array_equal(self.maze_view.robot, self.maze_view.goal):
        reward = 1
        done = True
    else:
        reward = -0.1/(self.maze_size[0]*self.maze_size[1])
        done = False
    return reward, done

# reward function v1
def calculate_reward_xy_1(self):

    
    # fixed reward for picking up an package. while less than arriving at terminal location
    if self.maze_view.maze.is_pick( tuple(self.maze_view.robot) ):
        self.maze_view.maze.load_pick( tuple(self.maze_view.robot) )
        reward = 1 / self.maze_view.maze.num_picks
        done = False

    # fixed reward for arriving terminal location
    elif np.array_equal(self.maze_view.robot, self.maze_view.goal):
        reward = 1
        done = True
        print("******************compeleted******************")
        print("loaded packages: ", self.maze_view.maze.get_loaded_picks())
    
    # fixed negative reward per step
    else:
        reward = -0.1/(self.maze_size[0]*self.maze_size[1])
        done = False

    return reward, done


# reward function v2
def calculate_reward_xy_2(self):


    # fixed reward for picking up an package. while less than arriving at terminal location
    if self.maze_view.maze.is_pick( tuple(self.maze_view.robot) ):
        self.maze_view.maze.load_pick( tuple(self.maze_view.robot) )
        reward = 1 / self.maze_view.maze.num_picks
        done = False

    # fixed reward for arriving at terminal location
    # based on how many packages been loaded
    elif np.array_equal(self.maze_view.robot, self.maze_view.goal):
        reward = len(self.maze_view.maze.get_loaded_picks())
        done = True
        print("******************compeleted******************")
        print("Last reward", reward)


    # fixed negative reward per step
    else:
        reward = -0.1/(self.maze_size[0]*self.maze_size[1])
        done = False
    
    return reward, done
