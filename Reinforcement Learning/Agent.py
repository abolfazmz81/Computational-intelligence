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


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
