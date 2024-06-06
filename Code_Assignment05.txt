import collections
from heapq import heappush, heappop
from collections import defaultdict, deque



# Q1
# Q1

from collections import defaultdict, deque

class Graph:
    def __init__(self, grid):
        self.grid = grid
        self.size_y = len(grid)
        self.size_x = len(grid[0])
        self.adjacency_list = defaultdict(list)
        # creating adjacency list using logic provided
        for k in range(self.size_y * self.size_x):
            i = k % self.size_x
            j = k // self.size_x
            if self.grid[j][i] == 1 or self.grid[j][i] == 2:    # we are ignoring the points labelled 2. as they are infeccted
                self.adjacency_list[k] = []  # Using k as unique node identifier
                if i > 0 and (self.grid[j][i-1] == 1 or self.grid[j][i-1] == 2):
                    self.adjacency_list[k].append(k - 1)
                if i < self.size_x - 1 and (self.grid[j][i+1] == 1 or self.grid[j][i+1] == 2):
                    self.adjacency_list[k].append(k + 1)
                if j > 0 and (self.grid[j-1][i] == 1 or self.grid[j-1][i] == 2):
                    self.adjacency_list[k].append(k - self.size_x)
                if j < self.size_y - 1 and (self.grid[j+1][i] == 1 or self.grid[j+1][i] == 2):
                    self.adjacency_list[k].append(k + self.size_x)

    def BFS(self, node=0):
            discovered = {i: False for i in range(len(self.grid) * len(self.grid[0]))}  # Initialize discovered dictionary
            queue = deque([node])  # Initialize the queue with the starting node
            discovered[node] = True  # Mark the starting node as discovered
            result = []
            infected = False

            while queue:
                current_node = queue.popleft()  # Dequeue the next node
                
                # Convert current_node to x, y coordinates
                x = current_node % self.size_x
                y = current_node // self.size_x

                # Skip if the current node is infected
                if self.grid[y][x] == 2:
                    infected = True
                    return [], infected

                result.append([y, x])  # Append the current node coordinates to the result list
                
                # Iterate through the neighbors of the current node
                for neighbor in self.adjacency_list[current_node]:
                    if not discovered[neighbor]:
                        queue.append(neighbor)  # Enqueue the neighbor
                        discovered[neighbor] = True  # Mark the neighbor as discovered
            
            return result, infected

def findSafeResidents(grid):
    residents = [] 
    discovered = set()  # to store visited safe residents
    graph = Graph(grid) # Create an instance of the Graph class

    # iterate through the nodes
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x] == 1 and (y, x) not in discovered:    # If the current node is a safe resident and not yet visited
                bfs_result, infected_staus = graph.BFS(y * len(grid[0]) + x)    # visit the node and perform BFS starting from the current node
                if infected_staus == False:
                    residents.extend(bfs_result)    # add discovered residents to the results
                    # Mark all nodes in BFS result as visited
                    for node in bfs_result:
                        discovered.add(tuple(node))
    residents.sort()    # sort the results
    return residents



# Q2
from collections import defaultdict
from heapq import heappop, heappush

def reverse_roads(n, roads):
    # Define graph structure with forward and backward edges
    graph = defaultdict(dict)
    for source, destination in roads:
        if destination in graph[source] and graph[source][destination] == 1:
            graph[source][destination] = 0
            graph[destination][source] = 0
        else:
            graph[source][destination] = 0
            graph[destination][source] = 1
    
    # Dijkstra's algorithm with priority queue for minimum cost path
    visited = set()
    min_heap = [(0, 1)]  # (cost, current city)
    min_reversals = float('inf')

    while min_heap:
        cost, curr_city = heappop(min_heap)

        if curr_city in visited:
            continue

        if curr_city == n and cost <= min_reversals:
            min_reversals = cost
            break

        visited.add(curr_city)

        for neighbor, edge_cost in graph[curr_city].items():
            new_cost = cost + edge_cost
            heappush(min_heap, (new_cost, neighbor))

    return min_reversals if min_reversals != float('inf') else -1



# Q3

def findAllRecipes(recipes, ingredients, supplies):
    graph = defaultdict(list)  # Graph to store recipe dependencies
    in_degree = {recipe: 0 for recipe in recipes}  # Tracks the number of ingredients needed for each recipe

    # Build the dependency graph
    for i, recipe in enumerate(recipes):
        for ingredient in ingredients[i]:
            if ingredient not in supplies:  # Ingredient needs to be created from another recipe
                graph[ingredient].append(recipe)
                in_degree[recipe] += 1

    # Use BFS algorithm to find recipes with in-degree 0 (can be created directly)
    queue = [recipe for recipe, count in in_degree.items() if count == 0]
    result = []
    while queue:
        recipe = queue.pop(0)
        result.append(recipe)
        for dependent_recipe in graph[recipe]:
            in_degree[dependent_recipe] -= 1
            if in_degree[dependent_recipe] == 0:
                queue.append(dependent_recipe)

    return result


# Q4

def connectInMinCost(connections):
    def find_parent(node):
        if parent[node] != node:
            parent[node] = find_parent(parent[node])
        return parent[node]

    connections.sort(key=lambda x: x[2])  # Sort connections by maintenance cost
    total_cost = 0
    parent = list(range(len(connections) + 1))  # Initialize parent array

    for connection in connections:
        source, destination, cost = connection
        parent_source = find_parent(source)
        parent_destination = find_parent(destination)

        if parent_source != parent_destination:
            parent[parent_source] = parent_destination
            total_cost += cost

    return total_cost




# Q5

def busRouter(numRoutes, dependencies):
    # Create an adjacency list representation of the graph
    graph = {i: [] for i in range(numRoutes)}
    in_degree = {i: 0 for i in range(numRoutes)}

    for dep in dependencies:
        a, b = dep
        graph[b].append(a)
        in_degree[a] += 1

    # Perform topological sort using BFS
    queue = [node for node in in_degree if in_degree[node] == 0]
    visited = set(queue)
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

    # If all nodes are visited, return True
    return len(visited) == numRoutes




# Q6
from collections import defaultdict, deque
from copy import deepcopy

def gameOfThrones(n: int, friendships: list[int], queries: list[int]) -> list[int]:
    # initilization
    if n == 0:
        return []
    result = []
    
    # functions to add relationships, remove relationships and count relationships after killing vulnerable nodes
    # add relationships
    def add_friend(node1, node2, adjacency_list):
        adjacency_list[node1].append(node2)
        adjacency_list[node2].append(node1)
        return adjacency_list

    # remove relationships
    def remove_friend(node1, node2, adjacency_list):
        print("Removing friend:", node1, "-", node2)
        print("Before removal:")
        print(adjacency_list)
        adjacency_list[node1].remove(node2)
        adjacency_list[node2].remove(node1)
        print("After removal:")
        print(adjacency_list)
        return adjacency_list

    from collections import deque

    def BFS_splitter(adjacency_list):
        visited = {node: False for node in adjacency_list}  # Initialize visited dictionary
        vulnerables = []
        for node in adjacency_list:
            if len(adjacency_list[node]) == 0:
                visited[node] = True
            elif not visited[node]:   # if node is not visited, perform BFS and find vulnerable nodes
  # if node is not visited, perform BFS and find vulnerable nodes
                queue = deque([node])  # Initialize the queue with the starting node

                while queue:
                    vulnerable = node
                    current_node = queue.popleft()
                    for neighbor in adjacency_list[current_node]:
                        if not visited[neighbor]:
                            queue.append(neighbor)  # Enqueue the neighbor
                            visited[neighbor] = True
                            vulnerable = min(vulnerable, neighbor)
                vulnerables.append(vulnerable)

        if len(vulnerables) == 0:
            print("No vulnerable nodes found.")
            return len(adjacency_list)

        print("Vulnerable nodes:", vulnerables)
        
        # Delete the vulnerable nodes and their relationships, then increment count_dead
        for vulnerable in vulnerables:
            print("Removing vulnerable node:", vulnerable)
            for neighbor in adjacency_list[vulnerable][:]:  # Iterate over a copy of the list
                visited[neighbor] = False
                adjacency_list = remove_friend(vulnerable, neighbor, adjacency_list)
            del adjacency_list[vulnerable]

        print("Updated adjacency list after removing vulnerable nodes:", adjacency_list)
        
        # Recursively call the function with the updated adjacency list
        return BFS_splitter(adjacency_list)





    # create and maintain a dictionary to model the friendships
    current_friendships = defaultdict(list)
    for i in range(1, n+1):
        current_friendships[i] = []
    for i in range(len(friendships)):
        current_friendships = add_friend(friendships[i][0], friendships[i][1], current_friendships)


    instance = 0
    for i in range(len(queries)):
        query = queries[i]
        if query[0] == 1:   # add friend
            print("Adding friend:", query[1], "-", query[2])
            current_friendships = add_friend(query[1], query[2], current_friendships)

        elif query[0] == 2: # remove friend
            print("Removing friend:", query[1], "-", query[2])
            current_friendships = remove_friend(query[1], query[2], current_friendships)

        elif query[0] == 3: # count safe nodes after killing vulnerable nodes
            instance += 1
            if instance == 1: 
                print("Instance 1")
                first_friendships = deepcopy(current_friendships)
                result.append(BFS_splitter(first_friendships))
            elif instance == 2:
                print("Instance 2")
                second_friendships = deepcopy(current_friendships)
                result.append(BFS_splitter(second_friendships))
            elif instance == 3:
                print("Instance 3")
                third_friendships = deepcopy(current_friendships)
                result.append(BFS_splitter(third_friendships))
    
    
    return result

# Q7

def optimalSignal(times: list[list[int]], n: int, k: int) -> int:
    # Create an adjacency list to represent the graph
    graph = [[] for _ in range(n + 1)]
    for u, v, w in times:
        graph[u].append((v, w))
    
    # Initialize distances to all nodes as infinity
    distances = [float('inf')] * (n + 1)
    # Distance from the source node to itself is 0
    distances[k] = 0
    
    # Priority queue to store nodes based on their tentative distances
    pq = [(0, k)]
    
    while pq:
        # Pop the node with the smallest tentative distance
        curr_dist, node = heappop(pq)
        # If the current distance is greater than the known distance, ignore it
        if curr_dist > distances[node]:
            continue
        # Iterate through neighbors of the current node
        for neighbor, weight in graph[node]:
            # Calculate the distance to the neighbor through the current node
            new_dist = curr_dist + weight
            # If the new distance is shorter than the known distance, update it
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heappush(pq, (new_dist, neighbor))
    
    # Check if there are any unreachable nodes
    for i in range(1, n + 1):
        if distances[i] == float('inf'):
            return -1
    
    # Return the maximum distance from the source node to any other node
    return max(distances[1:])



# Q8

def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    x_root = find(parent, x)
    y_root = find(parent, y)

    if rank[x_root] < rank[y_root]:
        parent[x_root] = y_root
    elif rank[x_root] > rank[y_root]:
        parent[y_root] = x_root
    else:
        parent[y_root] = x_root
        rank[x_root] += 1

def connectAllPoints(points):
    n = len(points)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j, manhattan_distance(points[i], points[j])))

    edges.sort(key=lambda x: x[2])
    parent = [i for i in range(n)]
    rank = [0] * n
    total_cost = 0
    for edge in edges:
        x, y, cost = edge
        x_root = find(parent, x)
        y_root = find(parent, y)
        if x_root != y_root:
            total_cost += cost
            union(parent, rank, x_root, y_root)

    return total_cost
