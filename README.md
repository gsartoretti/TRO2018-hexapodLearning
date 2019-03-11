# Hexapod RL
Reinforcement learning on a series elastic actuated hexapod robot

### File list

- EpisodeLearning.py: Online distributed learning code for body stabilization
     on the hexapod robot.
- EpisodeLearning_Centralized.py: Online centralized learning code for body
    stabilization
- RLstabilize.py: Offline trained policy execution (distributed)
- RLstabilize_Centralized.py: Offline trained policy execution (centralized)

### Requirements
- Numpy
- Tensorflow
- matplotlib

If necessary, the file "requirements.txt" contains a pip freeze of the
current working project.

**Note: The module "hebiapi" is currently not publicly available, but it is
only used for sending and recieving low-level joint information to the
robot.
