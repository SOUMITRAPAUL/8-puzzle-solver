import numpy as np


class Node:
    def __init__(self, state, parent, action):
        self.state = state  # Current state of the puzzle
        self.parent = parent  # Parent node
        self.action = action  # Action taken to reach this state


class StackFrontier:
    def __init__(self):
        self.frontier = []  # Stack for DFS

    def add(self, node):
        self.frontier.append(node)  # Add node to frontier

    def contains_state(self, state):
        # Check if a state is in the frontier
        return any((node.state[0] == state[0]).all() for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0  # Check if the frontier is empty

    def remove(self):
        if self.empty():
            raise Exception("Empty Frontier")
        else:
            node = self.frontier[-1]  # Last node (DFS)
            self.frontier = self.frontier[:-1]  # Remove last node
            return node


class QueueFrontier(StackFrontier):
    def remove(self):
        if self.empty():
            raise Exception("Empty Frontier")
        else:
            node = self.frontier[0]  # First node (BFS)
            self.frontier = self.frontier[1:]  # Remove first node
            return node


class Puzzle:
    def __init__(self, start, startIndex, goal, goalIndex):
        self.start = [start, startIndex]  # Initial state and index of blank tile
        self.goal = [goal, goalIndex]  # Goal state and index of blank tile
        self.solution = None  # To store the solution path

    def neighbors(self, state):
        mat, (row, col) = state  # Unpack state
        results = []  # List to store neighbor states

        # Moving the blank tile up, down, left, or right
        if row > 0:  # Move Down (instead of Up)
            mat1 = np.copy(mat)
            mat1[row][col] = mat1[row - 1][col]
            mat1[row - 1][col] = 0
            results.append(('down', mat1[row - 1][col], [mat1, (row - 1, col)]))

        if col > 0:  # Move Right (instead of Left)
            mat1 = np.copy(mat)
            mat1[row][col] = mat1[row][col - 1]
            mat1[row][col - 1] = 0
            results.append(('right', mat1[row][col - 1], [mat1, (row, col - 1)]))

        if row < 2:  # Move Up (instead of Down)
            mat1 = np.copy(mat)
            mat1[row][col] = mat1[row + 1][col]
            mat1[row + 1][col] = 0
            results.append(('up', mat1[row + 1][col], [mat1, (row + 1, col)]))

        if col < 2:  # Move Left (instead of Right)
            mat1 = np.copy(mat)
            mat1[row][col] = mat1[row][col + 1]
            mat1[row][col + 1] = 0
            results.append(('left', mat1[row][col + 1], [mat1, (row, col + 1)]))

        return results

    def print_solution(self):
        if self.solution is None:
            print("No solution found.")
            return

        print("Start State:\n", self.start[0], "\n")
        print("Goal State:\n", self.goal[0], "\n")
        print("\nStates Explored: ", self.num_explored, "\n")
        print("Solution Steps:\n")

        # Print each step of the solution with the correct action
        for action, moved_tile, cell in zip(self.solution[0], self.solution[1], self.solution[2]):
            print(f"Move tile {moved_tile} {action}\n", cell[0], "\n")
        print(f"Goal Reached in {len(self.solution[0])} actions!!")

    def does_not_contain_state(self, state):
        # Check if a state is not in the explored set
        for st in self.explored:
            if (st[0] == state[0]).all():
                return False
        return True

    def solve(self):
        self.num_explored = 0  # Initialize explored count

        start = Node(state=self.start, parent=None, action=None)  # Start node
        frontier = QueueFrontier()  # Create frontier
        frontier.add(start)  # Add start node to frontier

        self.explored = []  # List to store explored states

        while True:
            if frontier.empty():
                raise Exception("No solution")

            node = frontier.remove()  # Remove node from frontier
            self.num_explored += 1  # Increment explored count

            # Check if we reached the goal
            if (node.state[0] == self.goal[0]).all():
                actions = []
                moved_tiles = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    # Get the moved tile's value
                    moved_tile_index = node.state[1]  # Current position of the blank tile
                    moved_tile_value = node.parent.state[0][moved_tile_index]  # The tile that was moved
                    moved_tiles.append(moved_tile_value)
                    cells.append(node.state)
                    node = node.parent  # Move to parent node
                actions.reverse()  # Reverse actions to get correct order
                moved_tiles.reverse()  # Reverse moved tiles
                cells.reverse()  # Reverse states
                self.solution = (actions, moved_tiles, cells)  # Store the solution
                return

            # Mark node as explored
            self.explored.append(node.state)

            # Add neighbors to the frontier
            for action, moved_tile, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and self.does_not_contain_state(state):
                    child = Node(state=state, parent=node, action=action)  # Create child node
                    frontier.add(child)  # Add child to frontier


def get_puzzle_input():
    print(f"Enter the start state (3x3 grid) row by row, separated by spaces. Use 0 for the blank tile.")
    puzzle = []
    for i in range(3):
        row = input(f"Row {i + 1}: ").strip().split()
        row = [int(x) for x in row]
        puzzle.append(row)

    # Convert to numpy array
    puzzle = np.array(puzzle)

    # Find the index of the blank tile (0)
    blank_index = tuple(np.argwhere(puzzle == 0)[0])

    return puzzle, blank_index


def is_solvable(puzzle):
    """Check if the 3x3 puzzle is solvable."""
    flattened = puzzle.flatten()
    flattened = flattened[flattened != 0]  # Remove the blank tile (0) for inversion counting
    inversions = 0

    for i in range(len(flattened)):
        for j in range(i + 1, len(flattened)):
            if flattened[i] > flattened[j]:
                inversions += 1

    # For a 3x3 puzzle, if inversions are even, the puzzle is solvable
    return inversions % 2 == 0


# Take user input for the start state
start, startIndex = get_puzzle_input()

# Fixed goal state
goal = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])  # Set to desired goal state
goalIndex = (0, 0)  # Index of the blank tile (0) in goal state

# Check if the puzzle is solvable
if is_solvable(start):
    print("Puzzle is solvable. Solving now...")
    puzzle = Puzzle(start, startIndex, goal, goalIndex)  # Create puzzle instance
    puzzle.solve()  # Solve the puzzle
    puzzle.print_solution()  # Print the solution
else:
    print("This puzzle configuration is unsolvable.")
