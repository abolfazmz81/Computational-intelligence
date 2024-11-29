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

    # Store the user profile data
    user_profiles.append({
        'user_id': user_id,
        'age': age,
        'gender': gender,
        'location': location,
        'interests': ", ".join(interests)
    })

# Write the data to a CSV file
with open('user_profiles.csv', mode='w', newline='') as file:
    fieldnames = ['user_id', 'age', 'gender', 'location', 'interests']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for profile in user_profiles:
        writer.writerow(profile)

print("User profiles CSV file has been generated.")

content_items = []

for content_id in range(1, 51):
    # Random content data
    title = f"The movie {content_id}"
    genre = random.sample(genres, random.randint(2, 4))
    popularity = round(random.uniform(0.1, 1.0), 2)  # Popularity score between 0.1 and 1.0
    duration = random.randint(60, 180) # Length of the movie
    price = round(random.uniform(5, 30), 2)

    # Store the content data
    content_items.append({
        'content_id': content_id,
        'title': title,
        'genre': genre,
        'popularity': popularity,
        'duration':duration,
        'price':price
    })

# Write the data to a CSV file
with open('available_content.csv', mode='w', newline='') as file:
    fieldnames = ['content_id', 'title', 'genre', 'popularity', 'duration','price']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for content in content_items:
        writer.writerow(content)

print("Available content CSV file has been generated.")
