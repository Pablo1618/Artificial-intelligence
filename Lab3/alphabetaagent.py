import copy
from exceptions import AgentException

class AlphaBetaAgent:
    def __init__(self, my_token='o'):
        self.my_token = my_token

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')

        possible_moves = connect4.possible_drops()
        best_score = float('-inf')
        best_move = None

        for move in possible_moves:
            connect4_copy = copy.deepcopy(connect4)
            connect4_copy.drop_token(move)
            
            score = self.min_max(connect4_copy, depth=3, maximizing=False, alpha=float('-inf'), beta=float('inf'))
            # zaczynamy od maximizing = False bo to drugi ruch (ruch przeciwnika)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def min_max(self, connect4, depth, maximizing, alpha, beta):
        if depth == 0 or connect4.game_over:
            return self.evaluate_board(connect4)

        if maximizing:
            best_eval = float('-inf')
        else:
            best_eval = float('inf')
        
        possible_moves = connect4.possible_drops()

        for move in possible_moves:
            connect4_copy = copy.deepcopy(connect4)
            connect4_copy.drop_token(move)
            eval = self.min_max(connect4_copy, depth - 1, not maximizing, alpha, beta)
            # not maximizing - zmiana wartosci na przeciwna tak aby przeciwnik minimalizowal nasz wynik a nie maksymalizowal

            if maximizing:
                best_eval = max(best_eval, eval)
                alpha = max(alpha, best_eval)
            else:
                best_eval = min(best_eval, eval)
                beta = min(beta, best_eval)

            if beta <= alpha:
                break  # przyciecie aby nie szukac dalej niepotrzebnie
                # i tak wybierzemy drugi wierzcholek a nie ten

        return best_eval

    def evaluate_board(self, connect4):
        if connect4.wins == self.my_token:
            return 100
        elif connect4.wins != None:
            return -100
        else:
            if(self.my_token == 'x'):
                bot_token = 'x'
                opponent_token = 'o'
            else:
                bot_token ='o'
                opponent_token = 'x'
            bot_score = self.calculate_score(connect4, bot_token)
            opponent_score = self.calculate_score(connect4, opponent_token)
            return bot_score - opponent_score

    def calculate_score(self, connect4, token):
        score = 0
        for four in connect4.center_column():
            count = sum(1 for cell in four if cell == token)
            score += count
        for four in connect4.iter_fours():
            count = sum(1 for cell in four if cell == token)
            score += count
        return score
