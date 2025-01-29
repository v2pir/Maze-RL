# Maze-RL

**Agent**
Creates an agent to reward or penalize the robot for making specific moves at specific actions
Learns what actions need to be made through the DQN class
Epsilon decreases exponentially at a rate of 0.99995 (can be changed but a number above 0.99 is recommended for this simulation)

**DQN**
DQN algorithm
Calculates the correct action

**Environment**
Randomly generates a maze environment with the starting position being in the top left corner and the destination is either a point on the right or bottom edges
