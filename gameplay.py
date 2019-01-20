import random
import copy

class TicTacToe3X3GamePlay:

    def __init__(self):

        # current state stores how 'X's and 'O's are placed on the board
        self.currentState = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

        # 3 in a row, 3 in a column, 2 diagonals
        self.winningCombintation = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

        # some positions for rule-based bot to move on
        self.cornerPosition = [0, 2, 6, 8]
        self.middlePosition = 4
        self.allPosition = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # number to identity player X (DQN) and player O (rule-based)
        self.playerXMark = 1.0
        self.playerOMark = 0.0

    # new game by reset the board
    def ResetGame (self):
        self.currentState = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        return self.currentState

    # Player X proceed one step, then Player O
    def ProceedGameGivenAction(self, playerXAction):

        gameFinish = False
        result = "No Result"

        # if Player X make an invalid move with the given action
        if not self.IsBoardFreeOfSpaceGivenAction(self.currentState, playerXAction):
            return None, "Invalid Move", True

        # Player X proceeds with the action
        self.ApplyActionToBoard (self.currentState, playerXAction, self.playerXMark)

        # if player X wins
        if(self.IsWinnerGivenPlayer(self.currentState, self.playerXMark)):
            result = "Win"
            gameFinish = True

        # if a draw game
        elif self.IsFullGame():
            result = "Draw"
            gameFinish = True

        # if the game is not finished yet
        else:
            # player O proceeds
            playerOAction = self.PlayerOProccedOneStep()

            # Player O proceeds with the action
            self.ApplyActionToBoard (self.currentState, playerOAction, self.playerOMark)

            # if player O wins
            if (self.IsWinnerGivenPlayer(self.currentState, self.playerOMark)):
                result = "Lose"
                gameFinish = True

            # if a draw game
            elif self.IsFullGame():
                result = "Draw"
                gameFinish = True

        return self.currentState, result, gameFinish

    # find out if player X/O is winner
    def IsWinnerGivenPlayer(self, currentState, playerMark):
        # check all winning combinations
        for combination in self.winningCombintation:
            if (currentState[combination[0]] == currentState[combination[1]] == currentState[combination[2]] == playerMark):
                return True
        return False

    # rule-based Player O strategy
    def PlayerOProccedOneStep (self):
        # check if Player O can win in the next move
        for i in range(0,len(self.currentState)):
            currentStateCopy = copy.deepcopy(self.currentState)
            if self.IsBoardFreeOfSpaceGivenAction(currentStateCopy, i):
                self.ApplyActionToBoard (currentStateCopy, i, self.playerOMark)
                if self.IsWinnerGivenPlayer(currentStateCopy, self.playerOMark):
                    return i
        # check if Player X can win in the next move
        for i in range(0,len(self.currentState)):
            currentStateCopy = copy.deepcopy(self.currentState)
            if self.IsBoardFreeOfSpaceGivenAction(currentStateCopy, i):
                self.ApplyActionToBoard (currentStateCopy, i, self.playerXMark)
                if self.IsWinnerGivenPlayer(currentStateCopy, self.playerXMark):
                    return i

        # rule priority (1) take the corner (2) take the middle (3) take the rest randomly
        # (1) take the corner
        action = self.RandomNextAction(self.cornerPosition)
        if action != None:
            return action
        # (2) take the middle
        if self.IsBoardFreeOfSpaceGivenAction(self.currentState, self.middlePosition):
            return self.middlePosition
        # (3) take the rest randomly
        return self.RandomNextAction(self.allPosition)

    # check the board is free of space for the action
    def IsBoardFreeOfSpaceGivenAction (self, currentState, action):
        return currentState[action] == -1.0

    # check if the board is full
    def IsFullGame(self):

        for i in range(1,9):
            if self.IsBoardFreeOfSpaceGivenAction(self.currentState, i):
                return False
        return True

    def ApplyActionToBoard (self, currentState, index, move):
        currentState[index] = move

    # random pick the next action given positions
    def RandomNextAction (self, positionList):
        possibleActions = []
        # find all possible action
        for index in positionList:
            if self.IsBoardFreeOfSpaceGivenAction(self.currentState, index):
                possibleActions.append(index)
        # randomly choose one action from possibleActions
        if len(possibleActions) != 0:
            return random.choice(possibleActions)
        else:
            return None