"""
   Adrian Matyszczak - s19850
   Michał Kaczmarek - s18464

   System rekomendacji filmów,
   Wpisujemy dane użytkownika i szukamy dla niego drugiego najbardziej trafnego z ocenami filmów,
   System podaje pięć polecanych, nieobejrzanych filmów dla wpisanego użytkownika
   oraz pięć nie rekomendowanych filmów przez drugiego użytkownika.
"""
import numpy as np

'''
    Funkcja oblicza wynik odległości euklidesowej między użytkownikiem 1 a użytkownikiem 2
'''

def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    common_movies = {}
    # Filmy ocenione zarówno przez użytkownika 1, jak i użytkownika 2
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # Jeśli nie ma wspólnych filmów między użytkownikami zwracamy score = 0
    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))
