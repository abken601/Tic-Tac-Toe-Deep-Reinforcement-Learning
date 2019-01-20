# Tic-Tac-Toe-Deep-Reinforcement-Learning

Though the reinforcement learing problem for Tic-Tac-Toe is solved using Q-learning in one of my repository https://github.com/abken601/Tic-Tac-Toe-Reinforcement-Learning, it would be interesting to see what happens if we use deep reinforcement learning. In other word, how a network of much less parameters than the number of game states perform. Does the different combination of mark 'X' and 'O' are well described in the compressed network? 

In this experiment, we set up a 2-layer network and 9 outputs. They correspond to the Q functions of 9 possible actions on a 3x3 board. Be cautioned that invalid move (placing mark on an already occupied position) happens very frequently. This is much severely punished than losing a game! We define an reward for invalid move be -2, lose be -1, draw be 0 and win be 1. The network is called deep Q network, because the output values are Q functions for different actions and the back propagating gradients are going to improve the Q functions. We train a strategy for player X only (first move is done by player X), for player O we set a rule-based strategy and wish that our trained strategy can at least give a draw game against player O. 

We train our DQN by 200,000 episodes. In every 1,000 episodes, we use the 'trained' network to play 1,000 games and collect some stats, the result is as follows.

Round 1000
Invalid Move: 48.5 %, Win: 0.0 %, Draw: 0.0 %, Lose: 51.5 %

Round 2000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 50.5 %, Lose: 49.5 %

Round 3000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 52.4 %, Lose: 47.6 %

Round 4000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 47.3 %, Lose: 52.7 %

Round 5000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 51.9 %, Lose: 48.1 %

Round 6000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 51.7 %, Lose: 48.3 %

Round 7000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 49.8 %, Lose: 50.2 %

Round 8000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 52.1 %, Lose: 47.9 %

Round 9000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 51.7 %, Lose: 48.3 %

Round 10000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 50.1 %, Lose: 49.9 %

Round 11000
Invalid Move: 12.1 %, Win: 10.8 %, Draw: 53.2 %, Lose: 23.9 %

Round 12000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 74.8 %, Lose: 25.2 %

Round 13000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 75.4 %, Lose: 24.6 %

Round 14000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 49.3 %, Lose: 50.7 %

Round 15000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 77.8 %, Lose: 22.2 %

Round 16000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 24.9 %, Lose: 75.1 %

Round 17000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 73.3 %, Lose: 26.7 %

Round 18000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 74.8 %, Lose: 25.2 %

Round 19000
Invalid Move: 13.0 %, Win: 0.0 %, Draw: 14.5 %, Lose: 72.5 %

Round 20000
Invalid Move: 14.5 %, Win: 0.0 %, Draw: 12.6 %, Lose: 72.9 %

Round 21000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 26.0 %, Lose: 74.0 %

Round 22000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 22.0 %, Lose: 78.0 %

Round 23000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 0.0 %, Lose: 100.0 %

Round 24000
Invalid Move: 13.6 %, Win: 0.0 %, Draw: 34.4 %, Lose: 52.0 %

Round 25000
Invalid Move: 12.8 %, Win: 0.0 %, Draw: 36.8 %, Lose: 50.4 %

Round 26000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 0.0 %, Lose: 100.0 %

Round 27000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 25.2 %, Lose: 74.8 %

Round 28000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 26.2 %, Lose: 73.8 %

Round 29000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 23.7 %, Lose: 76.3 %

Round 30000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 23.9 %, Lose: 76.1 %

Round 31000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 23.4 %, Lose: 76.6 %

Round 32000
Invalid Move: 0.0 %, Win: 12.7 %, Draw: 0.0 %, Lose: 87.3 %

Round 33000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 26.5 %, Lose: 73.5 %

Round 34000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 22.7 %, Lose: 77.3 %

Round 35000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 50.3 %, Lose: 49.7 %

Round 36000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 25.3 %, Lose: 74.7 %

Round 37000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 25.9 %, Lose: 74.1 %

Round 38000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 24.8 %, Lose: 75.2 %

Round 39000
Invalid Move: 0.0 %, Win: 0.0 %, Draw: 25.0 %, Lose: 75.0 %

Round 40000
Invalid Move: 26.1 %, Win: 0.0 %, Draw: 25.2 %, Lose: 48.7 %
Despite a disappointing result, you may run the above simulation from main.py in the repository. We will improve the experiment by introducing DDQN (Deep Double Q-Learning). DDQN consists of two DQN, one is the target network to be trained, the other network generates behavior to improve the target network. After a certain episode, the behavior network is synchronized to the target network. This can prevent the overestimating behavior of just DQN. By decoupling the estimation using two networks, the target network can learn which states are valuable without interfered by the immediate next action estimated by itself. 
