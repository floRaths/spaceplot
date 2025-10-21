import random
import string

import numpy as np
from sklearn.datasets import make_blobs


def generate_fantasy_words(n, min_length=5, max_length=15):
    vowels = 'aeiou'
    consonants = ''.join(set(string.ascii_lowercase) - set(vowels))

    def create_word():
        length = random.randint(min_length, max_length)
        word = []

        # Ensure a mix of vowels and consonants, starting with a consonant
        for i in range(length):
            if i % 2 == 0:
                word.append(random.choice(consonants))
            else:
                word.append(random.choice(vowels))

        # Capitalize the first letter to make it look more like a name
        return ''.join(word).capitalize()

    return [create_word() for _ in range(n)]


def make_datasets(size=175000, n_cats=10, min_length=5, max_length=10):
    def make_centers(n):
        def make_number():
            coin = round(random.random() * 10) % 2

            rnd = random.random()

            if coin == 0:
                rnd = rnd * -1
            elif coin == 1:
                pass

            return round(rnd, 3)

        centers = []
        for i in range(n):
            center_coords = []
            center_coords.append(make_number())
            center_coords.append(make_number())

            centers.append(center_coords)

        return centers

    def scale_columns_to_range(arr, min_value=-2.5, max_value=2.5):
        # Select the first two columns
        cols = arr[:, :2]

        # Normalize the first two columns to the range [0, 1]
        min_cols = np.min(cols, axis=0)
        max_cols = np.max(cols, axis=0)
        normalized_cols = (cols - min_cols) / (max_cols - min_cols)

        # Scale the normalized columns to the desired range [-2.5, 2.5]
        scaled_cols = normalized_cols * (max_value - min_value) + min_value

        # Replace the original first two columns with the scaled ones
        arr[:, :2] = scaled_cols
        return arr

    centers = make_centers(n_cats)

    X, labels = make_blobs(n_samples=size, centers=centers, cluster_std=0.25, random_state=40)

    X = scale_columns_to_range(X, min_value=-2.25, max_value=2.25)
    x = X[:, 0]
    y = X[:, 1]

    fruits = generate_fantasy_words(n=n_cats, min_length=min_length, max_length=max_length)
    labels = np.array(fruits)[labels]

    category_data = np.column_stack((x, y, labels))

    rnd_values = np.random.negative_binomial(n=10, p=0.5, size=size)
    continuous_data = np.column_stack((x, y, rnd_values))

    return continuous_data, category_data
