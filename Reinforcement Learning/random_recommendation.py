import csv
import random


# Load user profiles from the user profiles CSV
def load_user_profiles(filename='user_profiles.csv'):
    user_profiles = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Parse user interests
            interests = row['interests'].split(', ')
            user_profiles.append({
                'user_id': int(row['user_id']),
                'interests': interests
            })
    return user_profiles


# Load available content from the available content CSV
def load_content_data(filename='available_content.csv'):
    content_data = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            content_data.append({
                'content_id': int(row['content_id']),
                'genre': row['genre']
            })
    return content_data


# Define possible interactions
interactions = ['watched', 'liked', 'disliked', 'skipped']


# Simulate interactions for users and content
def generate_interactions(user_profiles, content_data):
    interactions_data = []

    for user in user_profiles:
        # Step 1: Select 2 random content items for recommendations
        recommended_content = random.sample(content_data, 2)

        for content in recommended_content:
            # Step 2: Check genre match between content and user interests
            matching_genres = [genre for genre in user['interests'] if genre in content['genre']]

            if len(matching_genres) == 0:
                # If no genre match, the user either dislikes or skips
                interaction_type = random.choice(['disliked', 'skipped'])
            elif len(matching_genres) == 1:
                # If 1 matching genre, user is likely to watch, might like, dislike or skip
                interaction_type = random.choice(['watched', 'liked', 'skipped', 'disliked'])
            elif len(matching_genres) >= 2:
                # If 2 or more genres match, the user will most likely like or watch, sometimes dislike
                interaction_type = random.choice(['watched', 'liked', 'watched', 'liked', 'disliked'])

            # Add the interaction data
            interactions_data.append({
                'user_id': user['user_id'],
                'content_id': content['content_id'],
                'interaction': interaction_type
            })

    return interactions_data


# Save the interactions to a new CSV file
def save_interactions_to_csv(interactions_data, filename='user_content_interactions.csv'):
    with open(filename, mode='w', newline='') as file:
        fieldnames = ['user_id', 'content_id', 'interaction']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for interaction in interactions_data:
            writer.writerow(interaction)


# Main function to generate and save interactions
def main():
    # Load user profiles and content data
    user_profiles = load_user_profiles('user_profiles.csv')
    content_data = load_content_data('available_content.csv')

    # Generate interactions
    interactions_data = generate_interactions(user_profiles, content_data)

    # Save interactions to CSV
    save_interactions_to_csv(interactions_data, 'user_content_interactions.csv')

    print("User content interactions CSV file has been generated.")


# Run the main function
if __name__ == '__main__':
    main()
