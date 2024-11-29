import csv
import random
import faker
from random import choice

# Create a Faker instance
fake = faker.Faker()

# Sample genres for user interests
genres = ['action', 'comedy', 'drama', 'romance', 'adventure', 'thriller', 'horror', 'sci-fi', 'fantasy', 'documentary', 'anime', 'disturbing']

# Sample locations (cities and countries)
locations = ['New York, USA', 'London, UK', 'Toronto, Canada', 'Sydney, Australia', 'Berlin, Germany', 'Tokyo, Japan',
             'Paris, France', 'Mexico City, Mexico', 'Mumbai, India', 'Cape Town, South Africa','Tehran, Iran',
             'Madrid, Spain']

user_profiles = []

for user_id in range(1, 101):
    # Random demographic data
    age = random.randint(18, 65)
    gender = choice(['M', 'F'])
    location = choice(locations)

    # Random genres for user interests (2-5 genres per user)
    interests = random.sample(genres, random.randint(2, 5))

    # Simulate some interaction history
    interactions = []
    for content_id in range(1, 6):  # Simulate 5 content interactions(Fake)
        feedback = choice(['liked', 'disliked', 'watched', 'skipped'])
        interactions.append(f"{content_id}:{feedback}")

    # Store the user profile data
    user_profiles.append({
        'user_id': user_id,
        'age': age,
        'gender': gender,
        'location': location,
        'interests': ", ".join(interests),
        'interaction_history': "; ".join(interactions)
    })

# Write the data to a CSV file
with open('user_profiles.csv', mode='w', newline='') as file:
    fieldnames = ['user_id', 'age', 'gender', 'location', 'interests', 'interaction_history']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for profile in user_profiles:
        writer.writerow(profile)

print("User profiles CSV file has been generated.")
