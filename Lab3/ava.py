from exceptions import GameplayException
from connect4 import Connect4
from randomagent import RandomAgent
from minmaxagent import MinMaxAgent
from alphabetaagent import AlphaBetaAgent
import time

num_games = 1
alpha_beta_wins = 0
total_time = 0
start_time = time.time()
alpha_beta_token = 'o'
opponent_token = 'x'

for i in range(num_games):
    connect4 = Connect4(width=7, height=6)

    #agent1 = RandomAgent('o')

    agent1 = MinMaxAgent(opponent_token)

    agent2 = AlphaBetaAgent(alpha_beta_token)

    print("Game nr. " + str(i+1) + ":")

    while not connect4.game_over:
        connect4.draw()
        try:
            if connect4.who_moves == agent1.my_token:
                n_column = agent1.decide(connect4)
            else:
                n_column = agent2.decide(connect4)
            connect4.drop_token(n_column)
        except (ValueError, GameplayException):
            print('Invalid move')

    if connect4.wins == alpha_beta_token:
        alpha_beta_wins += 1
    
    connect4.draw()

end_time = time.time()
total_time = end_time - start_time

print("Alpha-beta minmax won", alpha_beta_wins, "/", num_games, "games")
print("Simulation time:", total_time, "seconds")
