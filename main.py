"""Adrian Matyszczak - s19850
   Michał Kaczmarek - s18464
   GRA NAI - stara chińska gra (nazywana tam Jianshizi, czyli 'gra w zabieranie kamieni') dla dwóch osób z użyciem 15 do 60 pionków.
   Zasady:
   -Pionki dzieli się na kupki dowolnej, różnej wielkości.
   -Kupek powinno być co najmniej trzy, i powinny zawierać co najmniej 4 pionki.
   -w każdej kupce powinna być inna liczba pionków.
   -Następnie gracze zabierają na zmianę dowolną, niezerową liczbę pionów.
   -W jednym ruchu wolno zbierać tylko z jednej kupki.
   - przegrywa gracz, który zabiera ostatni pionek.
"""

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax


class NIM(TwoPlayerGame):
    """Jest to podklasa klasy easyAI.TwoPlayersGame

    Methods:
        - __init__() : Inicjalizacja gry
        - possible_moves() : Wszystkie dozwolone ruchy w grze
        - make_move : Przekształcenie stosów zgodnie z ruchem gracza
        - win() : Zwraca warunki wygrania gry
        - show() : Wyświetlanie stanu gry
        - is_over() : Sprawdzanie, czy gra została zakończona
        - scoring() : Daje wynik w bieżącej grze
    """

    def __init__(self, gracze, max_remove=None):
        """ Inicjalizacja gry

        Parametry:
            - players : gracze
            - tab : Deklaracja wielkości stosów
            - current_player : Mówi, który gracz zaczyna
            - max_remove - maksymalna ilosc pionow zdjetych w jednej turze
        """
        self.players = gracze  # Przechowuje graczy w self.players (1 - Człowiek, 2 - AI)
        self.tab = [8, 12, 6]  # Początkowa lista elementów na stosach
        self.current_player = 2  # Zaczyna AI
        self.max_remove = max_remove

    def possible_moves(self):
        """ Wszystkie dozwolone ruchy w grze """
        # Gracz decyduje ile weźmie pionów ze stosu:
        return ["%d,%d" % (i + 1, j)
                for i in range(len(self.tab))   #wybor stosu
                for j in range(1, self.tab[i] + 1
                               if self.max_remove is None
                               else min(self.tab[i] + 1, self.max_remove) #ilosc pionow do zdjecia w turze
                               )
                ]

    def make_move(self, move):
        """ Przekształcenie stosów zgodnie z ruchem gracza """
        move = list(map(int, move.split(',')))
        # Druga wartość podana przez gracza (ilość pionów)
        # odejmujemy od pierwszej wartości (reprezentującej stos):
        self.tab[move[0] - 1] -= move[1]

    def win(self):
        """ Zwraca warunki wygranej/przegranej gry """
        # Warunkiem jest uzyskanie 0 w pierwszej, drugiej lub trzeciej stercie
        return self.tab[0] <= 0 or self.tab[1] <= 0 or self.tab[2] <= 0

    def is_over(self):
        """ Sprawdza, czy gra się zakończyła """
        return self.win()  # Gra kończy się, gdy ktoś wygrywa.

    def show(self):
        """ Wyświetla grę """
        print(self.tab)  # Aktualizuje status zmiennej stosu

    def scoring(self):
        """ Daje wynik w aktualnej grze """
        return 100 if self.win() else 0


# Rozpocznij mecz (i przechowuj historię ruchów po jego zakończeniu)
ai = Negamax(6)  # AI pomyśli o 6 ruchach z wyprzedzeniem
game = NIM([Human_Player(), AI_Player(ai)])  # szczegóły gry
history = game.play()  # uruchamia grę i inicjuje historię
""" historia zmiennych to lista [(g1,m1),(g2,m2)...] gdzie gi jest kopią
     gry po ruchu i, a mi jest ruchem wykonanym przez gracza
     czyja to była kolej.
"""
# wiadomosc ktory gracz wygral:
if game.current_player == 2:
    print('AI wygrało!')
else:
    print('Wygrałeś!')