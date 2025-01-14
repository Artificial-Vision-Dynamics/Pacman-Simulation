# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    path = [] # Path with actions to be returned
    explored = set() # Visited nodes 
    node = {'state': problem.getStartState(), 'cost': 0} # Node representation
    frontier = util.Stack()
    frontier.push((node, [])) # Push initial node and path
    
    if problem.isGoalState(node['state']): # Check if initial state is goal state
        return []
    while True:
        if frontier.isEmpty():
            raise Exception('Error in DFS')
        node,path = frontier.pop() # Pop node with path from frontier
        if node['state'] in explored: # Ignore explored nodes
            continue
        if problem.isGoalState(node['state']): # Reached our goal state
            return path
        successors = problem.getSuccessors(node['state']) # Get successors nodes
        for successor in successors:
            child_node = {'state': successor[0], 'cost': successor[2], 'parent_node': node, 'action': successor[1]}  
            new_path = path + [child_node['action']] # Update path
            frontier.push((child_node, new_path))        
        explored.add(node['state'])                                                             


def breadthFirstSearch(problem):

    """Search the shallowest nodes in the search tree first."""

    path = [] # Path with actions to be returned
    explored = set() # Visited nodes 
    node = {'state': problem.getStartState(), 'cost': 0} # Node representation
    frontier = util.Queue()
    frontier.push((node, [])) # Push initial node and path 
    
    if problem.isGoalState(node['state']): # Check if initial state is goal state
        return []
    while True:
        if frontier.isEmpty():
            raise Exception('Error in BFS')
        node,path = frontier.pop() # Pop node with path from frontier
        if node['state'] in explored: # Ignore explored nodes
            continue   
        if problem.isGoalState(node['state']): # Reached our goal state  
            return path
        successors = problem.getSuccessors(node['state']) # Get successors nodes
        for successor in successors:
            child_node = {'state': successor[0], 'cost': successor[2], 'parent_node': node, 'action': successor[1]}  
            new_path = path + [child_node['action']] # Update path
            frontier.push((child_node, new_path))        
        explored.add(node['state'])          
                         
def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    path = [] # Path with actions to be returned
    explored = set() # Visited nodes 
    node = {'state': problem.getStartState(), 'cost': 0} # Node representation
    frontier = util.PriorityQueue()
    frontier.push((node, []), node['cost']) # Push item: (node, path) based on priority:cost
    
    if problem.isGoalState(node['state']): # Check if initial state is goal state
        return []
    while True:
        if frontier.isEmpty():
            raise Exception('Error in UCS')
        node,path = frontier.pop()  # Take node and path
        if node['state'] in explored: # Ignore explored nodes
            continue
        if problem.isGoalState(node['state']): # Reached our goal state
            return path
        successors = problem.getSuccessors(node['state'])
        for successor in successors:
            child_node = {'state': successor[0], 'cost': successor[2], 'parent_node': node, 'action': successor[1]}  
            new_path = path + [child_node['action']] # Update path
            priority = child_node['parent_node']['cost'] + child_node['cost'] # calculate cost from root to child_node
            child_node['cost'] = priority # Update cost
            frontier.update((child_node, new_path), priority) # Update necessary changes in frontier if needed    
        explored.add(node['state'])    
                              

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    path = [] # Path with actions to be returned
    explored = set() # Visited nodes 
    node = {'state': problem.getStartState(), 'cost': 0}
    frontier = util.PriorityQueue()
    frontier.push((node, []), node['cost']) # Push item: (node, path) based on priority:cost
    
    if problem.isGoalState(node['state']):
        return []
    while True:
        if frontier.isEmpty():
            raise Exception('Error in A*')
        node,path = frontier.pop()  
        if node['state'] in explored: # Ignore explored nodes
            continue
        if problem.isGoalState(node['state']): # Reached our goal state
            return path
        successors = problem.getSuccessors(node['state'])
        for successor in successors:
            child_node = {'state': successor[0], 'cost': successor[2], 'parent_node': node, 'action': successor[1]}  
            new_path = path + [child_node['action']] # Update path
            new_cost = child_node['parent_node']['cost'] + child_node['cost']  # new_cost = Cost of root to child_node
            total_cost = new_cost + heuristic(child_node['state'], problem) # total_cost = new_cost + heuristic function cost
            child_node['cost'] = new_cost # Update cost
            frontier.update((child_node, new_path), total_cost) # Update necessary changes in frontier if needed       
        explored.add(node['state'])     
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
