import random
import numpy as np
import tensorflow as tf
from collections import deque

class DeepQNetwork():

    def __init__(self, session,
                 optimizer,
                 QNetworkStructure,
                 stateDimension,
                 actionQuantity):

        # tensorflow engine
        self.session = session
        self.optimizer = optimizer

        # network and replay buffer
        self.QNetworkStructure = QNetworkStructure
        self.replayBuffer = ReplayBuffer()

        # Q learning parameters
        self.batchSize = 32
        self.stateDimension = stateDimension
        self.actionQuantity = actionQuantity
        self.discountFactor = 1.0

        # training parameters
        self.maxGradient = 4.0
        self.regularizer = 0.01

        # counters
        self.saveReplayStepSize = 5
        self.saveExperienceCount = 0
        self.trainingCount = 0

        # create variables
        self.CreateQNetworkVariables()

        # intiialize variables
        self.session.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

        # assert all variables are initialized
        self.session.run(tf.assert_variables_initialized())

    # create variables for the Q network
    def CreateQNetworkVariables(self):

        # compute action from a state
        with tf.name_scope("predictActions"):

            # raw state representation
            self.states = tf.placeholder(tf.float32, (None, self.stateDimension), name="states")

            # initialize Q network
            with tf.variable_scope("QNetwork"):
                self.outputs = self.QNetworkStructure(self.states)

            # predict actions from Q network
            self.actionScores = tf.identity(self.outputs, name="actionScores")

            # create summary
            tf.summary.histogram("actionScores", self.actionScores)

            self.predicted_actions = tf.argmax(self.actionScores, axis=1, name="predictedActions")

        # estimate rewards using the next state: r(s_t,a_t) + argmax_a Q(s_{t+1}, a)
        with tf.name_scope("estimateCumulativeRewards"):

            # initialize array of state
            self.nextStates = tf.placeholder(tf.float32, (None, self.stateDimension), name="nextStates")

            # initialize array of state mask
            self.nextStateMasks = tf.placeholder(tf.float32, (None,), name="nextStateMasks")

            # initialize next output
            self.nextOutputs = self.QNetworkStructure(self.nextStates)

            # compute future rewards
            self.nextActionScores = tf.stop_gradient(self.nextOutputs)
            self.targetValues = tf.reduce_max(self.nextActionScores, reduction_indices=[1, ]) * self.nextStateMasks
            tf.summary.histogram("nextActionScores", self.nextActionScores)

            self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")

            # update reward with a discounted value
            self.cumulativeRewards = self.rewards + self.discountFactor * self.targetValues

        # compute gradients and loss
        with tf.name_scope("computeTemporalDifferences"):

            # compute temporal difference loss
            self.actionMasks = tf.placeholder(tf.float32, (None, self.actionQuantity), name="actionMask")
            self.maskedActionScores = tf.reduce_sum(self.actionScores * self.actionMasks, reduction_indices=[1, ])
            self.temporalDifference = self.maskedActionScores - self.cumulativeRewards
            self.temporalDifferenceLoss = tf.reduce_mean(tf.square(self.temporalDifference))

            # compute regularization loss
            q_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="QNetwork")
            self.regularizationLoss = self.regularizer * tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in q_network_variables])

            # compute gradients and total loss
            self.totalLoss = self.temporalDifferenceLoss + self.regularizationLoss
            gradients = self.optimizer.compute_gradients(self.totalLoss)

            # if gradient value larger than the max gradient, clip it
            for i, (gradient, variable) in enumerate(gradients):
                if gradient is not None:
                    gradients[i] = (tf.clip_by_norm(gradient, self.maxGradient), variable)

            # record gradients in the histograms
            for gradient, variable in gradients:
                tf.summary.histogram(variable.name, variable)
                if gradient is not None:
                    tf.summary.histogram(variable.name + '/gradients', gradient)

            # train the model by the gradient
            self.trainedOptimizer = self.optimizer.apply_gradients(gradients)

        self.noOptimizer = tf.no_op()

    def SaveExperience(self, state, action, result, nextState, gameFinish):

        # save the end state experience every saveReplayStepSize or when game is finish
        if self.saveExperienceCount % self.saveReplayStepSize == 0 or gameFinish:
            self.replayBuffer.AddExperienceToBuffer(state, action, result, nextState, gameFinish)

        # update counter
        self.saveExperienceCount += 1

    def EpsilonGreedyActionFromQNetwork(self, states):
        if random.random() <= 0.1:
            return random.randint(0, self.actionQuantity - 1)
        else:
            return self.session.run(self.predicted_actions, {self.states: states})[0]

    def BestActionFromQNetwork(self, states, ):
        return self.session.run(self.predicted_actions, {self.states: states})[0]

    def UpdateQNetwork (self):

        # if replay buffer does not have enough saved experience, return
        if self.replayBuffer.GetExperienceQuantity() < self.batchSize:
            return

        # initialize some parameters
        batch = self.replayBuffer.GetBatchFromBuffer(self.batchSize)
        states = np.zeros((self.batchSize, self.stateDimension))
        rewards = np.zeros((self.batchSize,))
        actionMasks = np.zeros((self.batchSize, self.actionQuantity))
        nextStates = np.zeros((self.batchSize, self.stateDimension))
        nextStateMasks = np.zeros((self.batchSize,))

        # get training result
        for stateIndex, (state, maskIndex, result, nextState, finish) in enumerate(batch):

            # update state array
            states[stateIndex] = state

            # update reward array
            if result == "Invalid Move":
                rewards[stateIndex] = -2.0
            elif result == "Lose":
                rewards[stateIndex] = -1.0
            elif result == "Win":
                rewards[stateIndex] = 10.0
            elif result == "Draw":
                rewards[stateIndex] = 0.0

            # reset mask
            actionMasks[stateIndex][maskIndex] = 1

            # when training finish, update next state array and next mask array
            if not finish:
                nextStates[stateIndex] = nextState
                nextStateMasks[stateIndex] = 1

        # perform one update of training
        self.session.run([self.totalLoss, self.trainedOptimizer, self.noOptimizer], \
                         {self.states: states, self.nextStates: nextStates, self.nextStateMasks: nextStateMasks, \
                          self.actionMasks: actionMasks, self.rewards: rewards})

        # update training counter
        self.trainingCount += 1

class ReplayBuffer():

    def __init__(self):

        # intiialize buffer parameters
        self.bufferSize = 10000
        self.experienceQuantity = 0
        self.buffer = deque()

    # random draw samples from the buffer
    def GetBatchFromBuffer (self, batchSize):
        return random.sample(self.buffer, batchSize)

    # add experience to buffer
    def AddExperienceToBuffer (self, state, action, result, nextAction, gameFinish):

        newExperience = (state, action, result, nextAction, gameFinish)

        # if buffer is not full, add new experience
        if self.experienceQuantity < self.bufferSize:
            self.buffer.append(newExperience)
            self.experienceQuantity += 1
        # if buffer is full, clear the buffer then add new experience
        else:
            self.buffer.popleft()
            self.buffer.append(newExperience)

    # get experience counter, max is bufferSize
    def GetExperienceQuantity (self):
        return self.experienceQuantity
