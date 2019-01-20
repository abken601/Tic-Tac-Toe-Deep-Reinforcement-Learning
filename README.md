# Tic-Tac-Toe-Deep-Reinforcement-Learning

Though the reinforcement learing problem for Tic-Tac-Toe is solved using Q-learning in one of my repository https://github.com/abken601/Tic-Tac-Toe-Reinforcement-Learning, it would be interesting to see what happens if we use deep reinforcement learning. In other word, how a network of much less parameters than the number of game states perform. Does the different combination of mark 'X' and 'O' are well described in the compressed network? 

In this experiment, we set up a 2-layer network and 9 outputs. They correspond to the Q functions of 9 possible actions on a 3x3 board. Be cautioned that invalid move (placing mark on an already occupied position) happens very frequently. This is much severely punished than losing a game! We define an reward for invalid move be -2, lose be -1, draw be 0 and win be 1. The network is called deep Q network, because the output values are Q functions for different actions and the back propagating gradients are going to improve the Q functions. We train a strategy for player X only (first move is done by player X), for player O we set a rule-based strategy and wish that our trained strategy can at least give a draw game against player O. 

We train our DQN by 200,000 episodes. In every 1,000 episodes, we use the 'trained' network to play 1,000 games and collect some stats, the result is as follows.


