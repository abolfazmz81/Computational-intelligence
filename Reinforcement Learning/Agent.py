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

    def update_q_value(self, user_id, content_id, interaction, next_content_ids):
        """ Update the Q-value based on the reward for the given user-content interaction. """
        reward = INTERACTION_REWARDS.get(interaction, 0)

        # Current Q-value for the user-content pair
        current_q_value = self.q_table[(user_id, content_id)][content_id]

        # Get the Q-values for the next content recommendations
        next_q_values = [self.q_table[(user_id, next_content_id)].get(next_content_id, 0) for next_content_id in
                         next_content_ids]

        # Calculate the max Q-value for the next recommended content
        max_future_q = max(next_q_values) if next_q_values else 0

        # Update the Q-value for the current (user_id, content_id) pair using the Q-learning update rule
        self.q_table[(user_id, content_id)][content_id] = current_q_value + ALPHA * (
                    reward + GAMMA * max_future_q - current_q_value)

    def train(self, interactions_data, epochs=100):
        """
        Train the Q-Learning agent using interaction data.
        """
        for epoch in range(epochs):
            for interaction in interactions_data:
                user_id = interaction['user_id']
                content_id = interaction['content_id']
                interaction_type = interaction['interaction']

                # Simulate recommending 2 content based on the past or randomly selected
                recommended_content = []
                for i in range(2):
                    # Epsilon chance for
                    recommended_content.append(
                        self.get_action(user_id, [content['content_id'] for content in self.content_data]))
                # recommended_content = random.sample(self.content_data, 2) # Alternative that uses completely random selection
                next_content_ids = [content for content in recommended_content]
                # Get the action (recommended content) and update Q-value based on user feedback
                self.update_q_value(user_id, content_id, interaction_type, next_content_ids)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} completed.")

    def recommend(self, user_id, content_ids):
        """
        Recommend the best content based on the Q-table for a given user.
        """
        return self.get_action(user_id, content_ids)

# Load user profiles and content data
def load_user_profiles(filename='user_profiles.csv'):
    user_profiles = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            interests = row['interests'].split(', ')
            user_profiles.append({'user_id': int(row['user_id']), 'interests': interests})
    return user_profiles


def load_content_data(filename='available_content.csv'):
    content_data = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            content_data.append({'content_id': int(row['content_id']), 'genre': row['genre']})
    return content_data


# Main function to initialize agent and train it
def main():
    user_profiles = load_user_profiles('user_profiles.csv')
    content_data = load_content_data('available_content.csv')

    # Initialize Q-learning agent
    agent = QLearningAgent(user_profiles, content_data)

    # Simulate training using the interaction data
    interactions_data = []
    with open('user_content_interactions.csv', mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            interactions_data.append({
                'user_id': int(row['user_id']),
                'content_id': int(row['content_id']),
                'interaction': row['interaction']
            })

    # Train the agent
    agent.train(interactions_data, epochs=100)



if __name__ == '__main__':
    main()
