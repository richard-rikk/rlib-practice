import gym
import math
import random
import numpy as np

from typing import Dict, Tuple, Any, Union

class CartPoleRightLeft(gym.Wrapper):
    """
    ### Observation Space
    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
    | 4   | Goal direction        | -1 for left         | 1 for right       |
    """
    def __init__(self, env_config:Dict[str,Any]) -> None:
      """
      Init class with the base environment.

      # Arguments
      - env_config:Dict[str,Any]
        Configuration information for the environment:
        "render_mode": "human" | None

      # Returns
      - None

      # Errors
      - None
      """
      super().__init__(gym.make("CartPole-v1", **env_config))
      
      
      high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                1,
            ],
            dtype=np.float32,
      )

      self.goal               = 1
      self.target_pos         = 2.0 # The target positions in +/-.
      self.target_tol         = 0.1 # The tolarance of error 0.1 = 10%.
      self.observation_space  = gym.spaces.Box(-high, high, dtype=np.float32)
    
    def __extend_state(self, state:np.ndarray) -> np.ndarray:
      """
      Appends the goal variable to the given state.
      # Arguments
      - state:np.ndarray
        A state of the environment.

      # Returns
      - The extended state.

      # Errors
      - None
      """
      return np.append(arr=state, values=self.goal)
    
    def __extend_reward(self, state:np.ndarray) -> float:
      """
      Changes the reward function according to the `goal` variable.

       # Arguments
      - state:np.ndarray
        A state of the environment.

      # Returns
      - reward: float
        The reward for the given state.

      # Errors
      - None
      """
      pos = state[0]
      return float(math.isclose(self.target_pos * self.goal, pos, rel_tol=self.target_tol))      

    def reset(self) -> np.ndarray:
      """
      Resets the environment.
      
      # Arguments
      - None

      # Returns
      - The starting state as a numpy array.

      # Errors
      - None
      """
      self.goal = -1 + 2*random.randint(0,1)
      state     = super().reset()
      return self.__extend_state(state=state)

    def step(self, action : Union[np.ndarray, int]) -> Tuple[np.ndarray, float, bool, Dict[str,Any]]:
      """
      Transforms state `s` with action `a` resulting in `s'`.

      # Arguments
      - action : np.ndarray | int
        The action to take in `s`. It can be either 0 or 1.

      # Returns
      - A tuple with the following:
        observation: np.ndarray
        reward: float
        done: bool

      # Errors
      - None
      """
      # np.array(self.state, dtype=np.float32), reward, terminated, False,
      state_, reward, terminated, _ = super().step(action=action)
      
      reward = self.__extend_reward(state=state_)
      state_ = self.__extend_state(state=state_)
      
      return state_, reward, terminated, {}