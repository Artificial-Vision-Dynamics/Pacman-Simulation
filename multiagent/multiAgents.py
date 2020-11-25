# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        totalScore = 0 # score to be returned 

        # First check if you can kill ghosts
        ghostIndex = 0
        ghostDistances = []
        
        for newGhostState in newGhostStates: # for every ghost
            if newScaredTimes[ghostIndex] != 0: # check if ghost is scared
                ghostDistance = manhattanDistance(newPos, newGhostState.getPosition())
                ghostDistances.append(ghostDistance)
            ghostIndex += 1    

        # Only if at least one ghost was scared
        if len(ghostDistances) != 0: 
            totalScore = -min(ghostDistances)       
            return totalScore              

        # Avoid situation where proposed position matches with ghost position        
        for newGhostState in newGhostStates:
            if newPos == newGhostState.getPosition():
                return float('-inf')

        # Calculating all distances from proposed position to all food coordinates and capsules
        # Taking minimum distance and returning the opposite number because  
        # our function returns a number, where higher numbers are better
        foodDistances = []
        foodAndCapsules = currentGameState.getFood().asList()
        foodAndCapsules += currentGameState.getCapsules() 
        currPosition = successorGameState.getPacmanPosition()

        for foodAndCapsulesCoordinate in foodAndCapsules:
            foodDistance = manhattanDistance(newPos, foodAndCapsulesCoordinate)
            foodDistances.append(foodDistance)
        totalScore = -min(foodDistances)    
        
        # A small penalty to Stop action
        if action == 'Stop':
            totalScore -= 10
            
        return totalScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimaxDecision(gameState):
            actions = gameState.getLegalActions(0) # actions of pacman
            maxValue = float('-inf') # set to worst possible value
            minimaxAction = None # initializing with nothing, its a legal action
            for a in actions: 
                pacmanState = gameState.generateSuccessor(0, a) # get next state of pacman
                pacmanValue = minValue(pacmanState, 0, 1) # call minVal from depth 0 and from 1st ghost
                if(pacmanValue > maxValue):  # find best action
                    maxValue = pacmanValue
                    minimaxAction = a
            return minimaxAction

        def maxValue(gameState, currDepth):
            if (gameState.isWin() == True) or (gameState.isLose() == True) or (currDepth == self.depth):
                return self.evaluationFunction(gameState)
            v = float('-inf') # set to worst possible value
            actions = gameState.getLegalActions(0) # actions of pacman
            for a in actions:
                pacmanState = gameState.generateSuccessor(0, a) # get next state of pacman
                v = max(v, minValue(pacmanState, currDepth, 1))
            return v
        
        def minValue (gameState, currDepth, agentIndex):
            if (gameState.isWin() == True) or (gameState.isLose() == True):
                return self.evaluationFunction(gameState)
            v = float('inf')  # set to worst possible value      
            totalNumOfGhosts = gameState.getNumAgents() - 1
            actions = gameState.getLegalActions(agentIndex) # actions of ghost
            for a in actions:
                ghostState = gameState.generateSuccessor(agentIndex, a) # get next state of ghost
                if agentIndex == totalNumOfGhosts: # its the last ghost so call maxVal with depth+1
                    v = min(v, maxValue(ghostState, currDepth + 1))
                else:
                    v = min(v, minValue(ghostState, currDepth, agentIndex + 1)) # call minVal for remaining ghosts
            return v    
        
        return minimaxDecision(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBetaSearch(gameState):
            actions = gameState.getLegalActions(0)
            # initialize with worst values
            maxValue = float('-inf')
            alpha = float('-inf')
            beta = float('inf')
            minimaxAction = None # initializing with nothing, its a legal action
            for a in actions:
                pacmanState = gameState.generateSuccessor(0, a) # next state of pacman
                pacmanValue = minValue(pacmanState, 0, 1, alpha, beta)
                if(pacmanValue > maxValue): # calculate best action 
                    maxValue = pacmanValue
                    minimaxAction = a
                alpha = max(alpha, maxValue)
            return minimaxAction

        def maxValue(gameState, currDepth, alpha, beta):
            if (gameState.isWin() == True) or (gameState.isLose() == True) or (currDepth == self.depth):
                return self.evaluationFunction(gameState)
            v = float('-inf') # set to worst possible value
            actions = gameState.getLegalActions(0) # actions of pacman
            for a in actions:
                pacmanState = gameState.generateSuccessor(0, a) # next state of pacman
                v = max(v, minValue(pacmanState, currDepth, 1, alpha, beta))
                if v > beta:  # no need to continue searching
                    return v
                alpha = max(alpha, v)
            return v
        
        def minValue (gameState, currDepth, agentIndex, alpha, beta):
            if (gameState.isWin() == True) or (gameState.isLose() == True):
                return self.evaluationFunction(gameState)
            v = float('inf')        
            totalNumOfGhosts = gameState.getNumAgents() - 1
            actions = gameState.getLegalActions(agentIndex) # actions of ghost
            for a in actions:
                ghostState = gameState.generateSuccessor(agentIndex, a) # get next state of ghost
                if agentIndex == totalNumOfGhosts: # its the last ghost so call maxVal with depth+1
                    v = min(v, maxValue(ghostState, currDepth + 1, alpha, beta))
                else:
                    v = min(v, minValue(ghostState, currDepth, agentIndex + 1, alpha, beta)) # call minVal for remaining ghosts
                if v < alpha: # no need to continue searching
                    return v    
                beta = min(beta, v)
            return v    
        
        return alphaBetaSearch(gameState)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimaxDecision(gameState):
            actions = gameState.getLegalActions(0) # actions of pacman
            maxValue = float('-inf') # set to worst possible value
            expectimaxAction = None # initializing with nothing, its a legal action
            for a in actions: 
                pacmanState = gameState.generateSuccessor(0, a) # get next state of pacman
                pacmanValue = expectedValue(pacmanState, 0, 1) # call minVal from depth 0 and from 1st ghost
                if(pacmanValue > maxValue):  # find best action
                    maxValue = pacmanValue
                    expectimaxAction = a
            return expectimaxAction

        def maxValue(gameState, currDepth):
            if (gameState.isWin() == True) or (gameState.isLose() == True) or (currDepth == self.depth):
                return self.evaluationFunction(gameState)
            v = float('-inf') # set to worst possible value
            actions = gameState.getLegalActions(0) # actions of pacman
            for a in actions:
                pacmanState = gameState.generateSuccessor(0, a) # get next state of pacman
                v = max(v, expectedValue(pacmanState, currDepth, 1))
            return v
        
        def expectedValue (gameState, currDepth, agentIndex):
            if (gameState.isWin() == True) or (gameState.isLose() == True):
                return self.evaluationFunction(gameState)
            v = 0      
            totalNumOfGhosts = gameState.getNumAgents() - 1
            actions = gameState.getLegalActions(agentIndex) # actions of ghost
            probability = 1.0 / len(actions) # uniform probability
            for a in actions:
                ghostState = gameState.generateSuccessor(agentIndex, a) # get next state of ghost
                if agentIndex == totalNumOfGhosts: # its the last ghost so call maxVal with depth+1
                    v += maxValue(ghostState, currDepth + 1)*probability  
                else: # call expectedValue for remaining ghosts
                    v += expectedValue(ghostState, currDepth, agentIndex + 1)*probability 
            return v # sum of each uniformly random value            
    
        return expectimaxDecision(gameState)
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    "*** YOUR CODE HERE ***"

    # if victory return infinity
    if currentGameState.isWin():
        return float("inf")

    # if defeat return -infinity 
    if currentGameState.isLose():
        return float("-inf")

    # usefull variables
    foodCoordinates = currentGameState.getFood().asList() 
    pacmanPosition = currentGameState.getPacmanPosition()  
    ghostStates = currentGameState.getGhostStates() 

    # evaluation parameters
    closestFood = 0
    closestScaredGhost = 0
    closestActiveGhost = 0
    numOfCapsules = len(currentGameState.getCapsules())
    numOfFood = len(foodCoordinates)

    # Calculating closest food distance
    foodDistances = []
    for foodCoordinate in foodCoordinates:
        foodDistance = manhattanDistance(pacmanPosition, foodCoordinate)
        foodDistances.append(foodDistance)
    closestFood = min(foodDistances)  

    # Calculating pacman distances from ghosts
    scaredGhostsDistances = []
    activeGhostsDistances = [] 

    for ghostState in ghostStates: # for every ghost state
        if ghostState.scaredTimer != 0: # if ghost is scared
            scaredGhostsDistance = manhattanDistance(pacmanPosition, ghostState.getPosition()) # calculate distance
            scaredGhostsDistances.append(scaredGhostsDistance)
        else: # if ghost is active
            activeGhostsDistance = manhattanDistance(pacmanPosition, ghostState.getPosition()) # calculate distance
            activeGhostsDistances.append(activeGhostsDistance)                 

    # calculating closest ghosts parameters in case coresponing lists are not empty
    if len(scaredGhostsDistances) > 0:
        closestScaredGhost = min(scaredGhostsDistances)
    if len(activeGhostsDistances) > 0:
        closestActiveGhost = min(activeGhostsDistances)   
 
    # every evaluation parameter has a specific weight
    # most important parameter is number of capsules remaining
    # logic is that the more important the parameter is the more points are subtracted from pacman
    # first goal of pacman should be getting capsules
    # second is eating all the food
    # third is getting away from active ghost
    # fourth getting to the closest food coordinate
    # fifth is hunting scared ghosts
    return -0.7*closestFood -0.3*closestScaredGhost - 1.5*closestActiveGhost - 20*numOfCapsules - 10*numOfFood

# Abbreviation
better = betterEvaluationFunction
