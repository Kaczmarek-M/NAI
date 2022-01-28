"""Adrian Matyszczak - s19850
   Michał Kaczmarek - s18464
   System szacowania potencjału zespołu siatkarskiego - Po wprowadzeniu danych poszczególnych zawodników z
   rankingu ogólnego zostanie obliczony współczynnik jakości zespołu.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

"""
    Za pomocą zmiennej poprzedzającej dla rozmytego systemu sterowania ustalamy pozycję w rankingu
    danego zawodnika, a za pomocą zmiennej następczej szacujemy potencjał zespołu. 
"""

receiver = ctrl.Antecedent(np.arange(0, 213, 1), 'receiver')
receiver2 = ctrl.Antecedent(np.arange(0, 213, 1), 'receiver2')
setter = ctrl.Antecedent(np.arange(0, 213, 1), 'setter')
middleBlocker = ctrl.Antecedent(np.arange(0, 213, 1), 'middleBlocker')
libero = ctrl.Antecedent(np.arange(0, 213, 1), 'libero')
attacker = ctrl.Antecedent(np.arange(0, 213, 1), 'attacker')
team_score = ctrl.Consequent(np.arange(0, 100, 1), 'team_score')

"""
    Wartości logi rozmytej przypisane do pozycji zawodnika w rankingu.
"""

receiver['low'] = fuzz.trimf(receiver.universe, [106, 159, 213])
receiver['medium'] = fuzz.trimf(receiver.universe, [53, 79, 106])
receiver['high'] = fuzz.trimf(receiver.universe, [0, 20, 53])

receiver2['low'] = fuzz.trimf(receiver2.universe, [106, 159, 213])
receiver2['medium'] = fuzz.trimf(receiver2.universe, [53, 79, 106])
receiver2['high'] = fuzz.trimf(receiver2.universe, [0, 20, 53])

setter['low'] = fuzz.trimf(setter.universe, [106, 159, 213])
setter['medium'] = fuzz.trimf(setter.universe, [53, 79, 106])
setter['high'] = fuzz.trimf(setter.universe, [0, 20, 53])

middleBlocker['low'] = fuzz.trimf(middleBlocker.universe, [106, 159, 213])
middleBlocker['medium'] = fuzz.trimf(middleBlocker.universe, [53, 79, 106])
middleBlocker['high'] = fuzz.trimf(middleBlocker.universe, [0, 20, 53])

libero['low'] = fuzz.trimf(libero.universe, [106, 159, 213])
libero['medium'] = fuzz.trimf(libero.universe, [53, 79, 106])
libero['high'] = fuzz.trimf(libero.universe, [0, 20, 53])

attacker['low'] = fuzz.trimf(attacker.universe, [106, 159, 213])
attacker['medium'] = fuzz.trimf(attacker.universe, [53, 79, 106])
attacker['high'] = fuzz.trimf(attacker.universe, [0, 20, 53])

"""
    Trzy zakresy jakości zespołu (low, medium, high)
"""

team_score['low'] = fuzz.trimf(team_score.universe, [50, 100, 100])
team_score['medium'] = fuzz.trimf(team_score.universe, [0, 50, 100])
team_score['high'] = fuzz.trimf(team_score.universe, [0, 0, 50])


"""
    Trzy zasady obliczania jakości zespołu po wprowadzeniu pozycji w rankingu zawodników
"""

rule1 = ctrl.Rule(
    receiver['low'] | receiver2['low'] | setter['low'] | middleBlocker['low'] | libero['low'] | attacker['low'],
    team_score['low'])
rule2 = ctrl.Rule(receiver['medium'] | receiver2['medium'] | setter['medium'] | middleBlocker['medium']
                  | libero['medium']
                  | attacker['medium'], team_score['medium'])
rule3 = ctrl.Rule(receiver['high'] | receiver2['high'] | setter['high'] | middleBlocker['high']
                  | libero['high']
                  | attacker['high'], team_score['high'])

"""
    Dostarczamy dane do klasy bazowej ControlSystem która zawiera rozmyty system sterowania
    a nasteępnie za pomocą ControlSystemSimulation obliczamy wynik z ControlSystem.
"""

team_assessment = ctrl.ControlSystem([rule1, rule2, rule3])
assessment = ctrl.ControlSystemSimulation(team_assessment)

print("System oceny zespolu PlusLigi")
print("podaj pozycje zawodnika w rankingu od 1 do 213")
print("atakujacy od 1 do 213")
receiver = input("przyjmujący: ")
receiver2 = input("drugi przyjmujacy: ")
setter = input("rozgrywajacy: ")
middleBlocker = input("środkowy: ")
libero = input("libero: ")
attacker = input("atakujący: ")
"""
    Użytkownik wpisuje dane 
"""
assessment.input['receiver'] = int(receiver)
assessment.input['receiver2'] = int(receiver2)
assessment.input['setter'] = int(setter)
assessment.input['middleBlocker'] = int(middleBlocker)
assessment.input['libero'] = int(libero)
assessment.input['attacker'] = int(attacker)

"""
    Obliczamy system rozmyty i podajemy wynik oraz wyświetlamy wykres logi rozmytej.
"""

assessment.compute()
print("team score = ", assessment.output['team_score'])
team_score.view(sim=assessment)
