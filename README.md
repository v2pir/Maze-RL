# Maze-RL

**Agent**
<br>
Creates an agent to reward or penalize the robot for making specific moves at specific actions
<br>
Learns what actions need to be made through the DQN class
<br>
Epsilon decreases exponentially at a rate of 0.99995 (can be changed but a number above 0.99 is recommended for this simulation)
<br>

**DQN**
<br>
DQN algorithm
<br>
Calculates the correct action
<br>

**Environment**
<br>
Randomly generates a maze environment
<br>
The starting position is the top left corner and the destination is either a point on the right or the bottom edges
