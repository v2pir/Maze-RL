import numpy as np
import pygame
import random


class Environment:
    def __init__(self, screen_dim, tile_size):
        self.elements = ["X", "O"]
        self.directions = ["LEFT", "UP", "RIGHT", "DOWN"]
        self.init_state = np.array([20, 20])

        pygame.init()
 
        # Window
        self.SCREEN_WIDTH = screen_dim[0]
        self.SCREEN_HEIGHT = screen_dim[1]
        self.SCREEN = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.FONT = pygame.font.SysFont('Futura', 30)
        pygame.display.set_caption('RL Map')

        # Colors
        self.GREY = (60, 60, 60)
        self.BG = (0, 0, 0)
        self.GREEN = (100, 255, 10)
        self.BLUE = (70, 130, 255)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)

        # Variables
        self.tile_size = tile_size
        self.visted = {}
        self.goal_state = None

    def createMap(self):
        centers = []
        for x in range(int(self.SCREEN_WIDTH / self.tile_size)):
            for y in range(int(self.SCREEN_HEIGHT / self.tile_size)):
                centers.append((20 + x * 40, 20 + y * 40))

        map = {centers[i]: "X" for i in range(len(centers))}

        pos = self.init_state.copy()
        map[tuple(pos)] = "O"

        c = centers.copy()

        while pos[0] != 780 and pos[1] != 780:
            # Find random direction
            dir = self.directions[random.randint(0, 3)]

            # Move accordingly
            if dir == "DOWN":
                pos[1] += 40
            if dir == "RIGHT":
                pos[0] += 40
            if dir == "LEFT":
                pos[0] -= 40
            if dir == "UP":
                pos[1] -= 40

            if pos[0] < 0:
                pos[0] += 40
                continue
            if pos[1] < 0:
                pos[1] += 40
                continue

            n = tuple(pos)
            if map[n] == "O":
                continue

            # Replace the "X" with "O"
            map[n] = "O"

            # Remove this coordinate from map so it doesn't get repeated
            c.remove(n)

        return centers, map
    
    def reset_run(self):
        return self.init_state

    def draw_grid(self, tile_size):
        self.SCREEN.fill(self.BG)

        # Draw vertical lines
        for x in range(tile_size, self.SCREEN_WIDTH, tile_size):
            pygame.draw.line(self.SCREEN, self.GREY, (x, 0), (x, self.SCREEN_HEIGHT))

        # Draw horizontal lines
        for y in range(tile_size, self.SCREEN_HEIGHT, tile_size):
            pygame.draw.line(self.SCREEN, self.GREY, (0, y), (self.SCREEN_WIDTH, y))

    def draw_map(self, centers, map):
        # Draw map tiles, walls, and open spaces
        for coord in centers:
            letter = map[coord]
            if letter == "O":
                color = self.BLUE
            else:
                color = self.RED
            img = self.FONT.render(letter, True, color)
            self.SCREEN.blit(img, np.array(coord) - np.array([14, 20]))

    def draw_robot(self, location):
        letter = "R"
        color = self.WHITE
        robot = self.FONT.render(letter, True, color)
        self.SCREEN.blit(robot, location)

    def remove_robot(self, location):
        letter = "R"
        color = self.BG
        robot = self.FONT.render(letter, True, color)
        self.SCREEN.blit(robot, location)

    # Draw grid
    def draw_environment(self, centers, map):
        self.draw_grid(self.tile_size)
        self.draw_map(centers, map)

    def draw_rewards(self, map):
        # Draw rewards for each tile
        cent = list(map.keys())
        reward_map = {cent[i]: -1 for i in range(len(cent))}

        self.visited = {coord: 0 for coord in map.keys()}

        return reward_map

    def step(self, current_pos, action, map):
        done = False
        reward = 0

        location = current_pos.copy()
        reward_map = self.draw_rewards(map)

        # Move accordingly
        if action == "DOWN":
            location[1] += 40
        if action == "RIGHT":
            location[0] += 40
        if action == "LEFT":
            location[0] -= 40
        if action == "UP":
            location[1] -= 40

        # Edge cases
        if location[0] < 0:
            location[0] += 40
            reward = -10
        if location[1] < 0:
            location[1] += 40
            reward = -10
        if location[0] > 780 or location[1] > 780:
            done = True
            return current_pos, location, 10000, done
        if map[tuple(location)] == "X":
            location = current_pos
            reward = -10
        else:
            reward = reward_map[tuple(location)]

        # Update step function
        if tuple(location) in self.visited:
            self.visited[tuple(location)] += 1
            reward -= self.visited[tuple(location)] * 0.5  # Penalize repeat visits
        else:
            reward += 12  # Reward finding new spots

        if self.goal_state is not None:
            distance_to_goal = np.linalg.norm(np.array(location) - np.array(self.goal_state))
            previous_distance = np.linalg.norm(np.array(current_pos) - np.array(self.goal_state))

            if distance_to_goal < previous_distance:
                reward += 8  # Reward progress
            if distance_to_goal > previous_distance:
                reward -= 5

        return current_pos, location, reward, done