from environment import Environment
from Agent import Agent
import pygame
import time
import numpy as np
 
env = Environment(screen_dim=[800,800], tile_size=40)

cent, map = env.createMap()
env.draw_environment(cent, map)
action_space = env.directions

agent = Agent(gamma=0.99, epsilon=1.0, batch_size=32, n_actions=4,
              eps_end=0.01, input_dims=[2], lr=0.003)

scores=[]
eps_history=[1]
n_games=500
robot_display_offset = np.array([-10,-19])

def displayEnv():
    done = False
    score = 0
    location = env.reset_run()

    while not done:
        env.remove_robot(location + robot_display_offset)
        action = agent.pick_action(location)
        action_move = action_space[action]
        prev_location, location, reward, done = env.step(location, action_move, map)
        reward += np.random.uniform(-0.1, 0.1)  # add slight randomness to rewards
        score += reward
        agent.store_transition(prev_location, action, reward, location, done)
    
        agent.learn()
        # time.sleep(0.2)
        env.draw_robot(location + robot_display_offset)

        if location[0] > 780 or location[1] > 780:
            if location[0] > 780:
                location[0] -= 40
                env.goal_state = location
            if location[1] > 780:
                location[1] -= 40
                env.goal_state = location
            done = True

        if score < -20000:
            done = True

        pygame.display.update()

        # Event handler
        for event in pygame.event.get():
            # Quit game
            if event.type == pygame.QUIT:
                done = True

    if score < -20000:
        env.remove_robot(location + robot_display_offset)
        average_score = np.mean(scores[-100:])
        agent.epsilon = eps_history[-1]
        return score, average_score, agent.epsilon

    scores.append(score)
    eps_history.append(agent.epsilon)
    average_score = np.mean(scores[-100:])

    return score, average_score, agent.epsilon

for reps in range(n_games):
    score, avg_score, eps = displayEnv()
    print("score:", score, "\naverage score:", avg_score, "\nepsilon:", eps)
    time.sleep(1)

pygame.quit()