"""
   Adrian Matyszczak - s19850
   Michał Kaczmarek - s18464

   System rekomendacji filmów,
   Wpisujemy dane użytkownika i szukamy dla niego drugiego najbardziej trafnego z ocenami filmów,
   System podaje pięć polecanych, nieobejrzanych filmów dla wpisanego użytkownika
   oraz pięć nie rekomendowanych filmów przez drugiego użytkownika.

"""
import json
from compute_scores import euclidean_score

'''
    Funkcja polecane / nie polecane filmy ,dla podanego użytkownika
'''


def getRecommendation(bestScore, userList):
    new_list = {}
    for key, value in userList.items():
        if key not in data[user].keys():
            new_list[key] = value
    sort_list = dict(sorted(new_list.items(), key=lambda element: element[1]))   # sortowanie listy
    suggestMovies = dict(list(sort_list.items())[-5:])                           # wstawianie 5 sugerowanych filmów
    notSuggestMovie = dict(list(sort_list.items())[:5])                          # wstawianie 5 nie polecanych filmów

    i = 0
    moviesList = []

    print('\n' + bestScore['user'] + ' dla ' + user + ' poleca obejrzeć:')       # wyświetlanie polecanych filmów
    for item in suggestMovies:
        print(item + ' - ' + 'ocena: ' + str(suggestMovies[item]))
        i += 1
        moviesList.append(item)

    print('\n' + bestScore['user'] + ' dla ' + user + ' nie poleca filmów:')     # wyświetlanie nie polecanych filmów
    for item in notSuggestMovie:
        print(item + ' - ' + 'ocena: ' + str(notSuggestMovie[item]))
        i += 1
        moviesList.append(item)


if __name__ == '__main__':

    print("Podaj użytkownika: ")
    user = input()

    score_type = 'EuclideanAlg.'

    ratings = 'userData.json'
    with open(ratings, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())
        '''
            Filtrowanie listy filmów, których nie oglądał user 2
        '''
    scoreList = []
    for item in data:
        if item != user:
            if score_type == 'EuclideanAlg.':
                scoreList.append({'score': euclidean_score(data, user, item), 'user': item})

    bestScore = max(scoreList, key=lambda x: x['score'])
    userList = data[bestScore['user']]
    moviesList = getRecommendation(bestScore, userList)
