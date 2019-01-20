import tensorflow as tf
import numpy as np
from DQN import DeepQNetwork
from gameplay import TicTacToe3X3GamePlay

# define parameters of a (128x64x9) network
def QNetworkStructure(states):

    W1 = tf.get_variable("W1", [stateDimension, 512], initializer=tf.random_normal_initializer(stddev=0.1))
    b1 = tf.get_variable("b1", [512], initializer=tf.constant_initializer(0))
    h1 = tf.nn.relu(tf.matmul(states, W1) + b1)
    W2 = tf.get_variable("W2", [512, 64], initializer=tf.random_normal_initializer(stddev=0.1))
    b2 = tf.get_variable("b2", [64], initializer=tf.constant_initializer(0))
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    Wo = tf.get_variable("Wo", [64, actionQuantity], initializer=tf.random_normal_initializer(stddev=0.1))
    bo = tf.get_variable("bo", [actionQuantity], initializer=tf.constant_initializer(0))
    probability = tf.matmul(h2, Wo) + bo

    return probability

# initialize game env
gameplay = TicTacToe3X3GamePlay()

# initialize tensorflow engine
sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00005, decay=0.9)

# define policy neural network for 3x3 game
stateDimension = 9
actionQuantity = 9
QNetwork = DeepQNetwork(sess, optimizer, QNetworkStructure, stateDimension, actionQuantity)

# iteration to train Q network
trainingIteration = 200001

# start training
reward = 0.0
for iRound in range(trainingIteration):

    # start a new game
  state = np.array(gameplay.ResetGame())

  # one game does not take more than 25 steps
  for step in range(25):

    # get the action from the Q network (epsilon greedy policy)
    action = QNetwork.EpsilonGreedyActionFromQNetwork(state[np.newaxis,:])

    # proceed the game with the action
    next_state, result, gameFinish = gameplay.ProceedGameGivenAction(action)

    # save the result
    QNetwork.SaveExperience(state, action, result, next_state, gameFinish)

    # update the model by back propagation
    QNetwork.UpdateQNetwork()

    # update the state
    state = np.array(next_state)

    # when the game is done, exit
    if gameFinish:
        break

  # play demo game every 100 rounds for intermediate report
  if iRound % 100 == 0:

    # counter for 4 different game results
    invalidMoveCount = 0
    winCount = 0
    drawCount = 0
    loseCount = 0

    # play demo 1000 times
    for j in range(1000):

        # start a new game
        state = np.array(gameplay.ResetGame())

        # one game does not take more than 25 steps
        for step in range(25):

            # get the action from the Q network (best policy)
            #action = QNetwork.EpsilonGreedyActionFromQNetwork(state[np.newaxis, :])
            action = QNetwork.BestActionFromQNetwork(state[np.newaxis, :])

            # proceed the game with the action
            next_state, result, gameFinish = gameplay.ProceedGameGivenAction(action)

            # update the state
            state = np.array(next_state)

            # when the game is done, update the counters
            if gameFinish:
                if result == "Invalid Move":
                    invalidMoveCount += 1
                elif result == "Lose":
                    loseCount += 1
                elif result == "Win":
                    winCount += 1
                elif result == "Draw":
                    drawCount += 1
                break

    # report demo game result
    print("Round", iRound)
    print("Invalid Move:", invalidMoveCount / 10.0, "%")
    print("Win:", winCount / 10.0, "%")
    print("Draw:", drawCount / 10.0, "%")
    print("Lose:", loseCount / 10.0, "%")
    print()

    # reset result count
    invalidMoveCount = 0
    winCount = 0
    drawCount = 0
    loseCount = 0