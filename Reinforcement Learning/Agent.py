# This is a sample Python script.
import random
import numpy as np
import csv

# Q-learning Attributes
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.2  # Exploration rate (epsilon-greedy)(0.4 for more exploration)

# Reward function
INTERACTION_REWARDS = {
    'watched': 1,  # Positive feedback
    'liked': 2,  # Ultra positive feedback
    'disliked': -1,  # Negative feedback
    'skipped': 0  # Neutral feedback
}


class QLearningAgent:
    def __init__(self, user_profiles, content_data):
        """
        Initialize Q-Table with zeros
        :param user_profiles: all the users
        :param content_data: all the movies
        """
        self.q_table = {}
        self.user_profiles = user_profiles
        self.content_data = content_data

        # Initialize Q-table with state-action pairs: (user_id, content_id)
        for user in user_profiles:
            for content in content_data:
                self.q_table[(user['user_id'], content['content_id'])] = {
                    action: 0 for action in range(1, len(content_data) + 1)  # Actions are content IDs
                }

    def get_action(self, user_id, content_ids):
        """
        Select an action using epsilon-greedy strategy:
        - With probability epsilon, choose a random action (exploration).
        - With probability (1 - epsilon), choose the best action (exploitation).
        """
        if random.uniform(0, 1) < EPSILON:
            return random.choice(content_ids)  # Exploration
        else:
            q_values = {content_id: self.q_table[(user_id, content_id)][content_id] for content_id in content_ids}
            return max(q_values, key=q_values.get)  # Exploitation


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
