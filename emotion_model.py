import random

# Placeholder stub for the emotion prediction function.
# Replace this with your actual model from the YouTube tutorial.
def predict_emotion(frame):
    # Dummy randomizer for testing purposes.
    labels = ["happy", "sad", "angry", "neutral"]
    label = random.choice(labels)
    confidence = round(random.uniform(0.5, 1.0), 2)
    return label, confidence
