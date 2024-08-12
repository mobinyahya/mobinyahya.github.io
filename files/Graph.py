from collections import defaultdict
import heapq

class Graph(object):
    def __init__(self):
        self.graph = defaultdict(list)

        # check if there is a cycle in a directed graph
        self.DAG = True



    # Function to add an edge to graph
    def addEdge(self, u, v, weight=0):
        if u not in self.graph:
            self.addNode(u)
        if v not in self.graph:
            self.addNode(v)
        # self.graph[u].append((v, weight)) # if weighted
        # self.graph[v].append((u, weight)) # if weighted
        self.graph[u].append(v)
        self.graph[v].append(u)

    def addNode(self, u):
        self.graph[u] = []

    def DFSUtil(self, v, visited, cycle_set):
        # Mark the current node as visited and print it
        visited.add(v)
        print(v, end=' ')

        # Cycle path set in a directed graph. Keep adding node as we get more depth
        # but remove the node when we backtrack up
        cycle_set.add(v)

        # Recur for all the vertices adjacent to this vertex
        for neighbour in self.graph[v]:
            if neighbour in cycle_set:
                self.DAG = False
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited, cycle_set)

        #  remove the node when we backtrack up,
        #  as we only care about directed cycles, not just any cycle
        cycle_set.remove(v)

    def BFS(self, s):
        # if you want to use bfs to find the edge distance to the source
        distances = {}
        distances[s] = 0  # Distance to itself is 0

        # Mark all the vertices as not visited
        visited = set()
        # Create a queue for BFS
        # queue = [(s, depth=0)] # if you want to capture the depth of each layer, add the tuple with depth to the queue.
        queue = []
        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited.add(s)

        while queue:
            # Dequeue a vertex from queue and print it
            s = queue.pop(0)
            print(s, end=" ")

            # Get all adjacent vertices of the dequeued vertex s.
            # If an adjacent has not been visited, then mark it visited and enqueue it
            for v in self.graph[s]:
                if v not in visited:
                    queue.append(v)
                    visited.add(v)
                    # queue.append((v, distances[s] + 1))
                    distances[v] = distances[s] + 1

    def CC_Count(self):
        cc_count = 0
        visited = set()
        for u in self.graph:
            if u not in visited:
                cc_count +=1
                self.DFSUtil(u, visited)
        return cc_count

    # --------------------------------------------------------
    # --------------------------------------------------------
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
        # Initialize minimum distance for next node
        min = 1e7

        # Search not nearest vertex not in the
        # shortest path tree
        min_index = -1
        for v in self.graph:
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v

        return min_index

    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, start_node):
        dist = {}
        sptSet = {}
        for node in self.graph:
            dist[node] = 1e7
            sptSet[node] = False
        dist[start_node] = 0

        for iter in range(len(self.graph)):
            # Pick the minimum distance vertex
            u = self.minDistance(dist, sptSet)
            if u == -1:
                # it means the remaining nodes are not connected
                return dist
            # Add min dist vertex to spt
            sptSet[u] = True

            # Update dist value of adjacent vertices of the picked vertex
            for v, weight in self.graph[u]:
                if sptSet[v] == False:
                    if dist[v] > dist[u] + weight:
                        dist[v] = dist[u] + weight
        return dist


    # --------------------------------------------------------
    # --------------------------------------------------------
    def bellman_ford(self, src):
        # for each node v: Predecessors of each node in the shortest path
        predecessor = {}
        # for each node v: Shortest path from source
        dist = {}
        for u in self.graph:
            dist[u] = float("Inf")
        # initialize the source vertex distance as 0
        dist[src] = 0

        # relax all edges in the graph (|V|-1) times
        for _ in range(len(self.graph) - 1):
            for s in self.graph:
                for d, w in self.graph[s]:
                    if dist[s] != float("Inf") and dist[s] + w < dist[d]:
                        dist[d] = dist[s] + w
                        predecessor[d] = s

        # Detect negative cycle
        # if value changes then we have a negative cycle in the graph
        for s in self.graph:
            for d, w in self.graph[s]:
                if dist[s] != float("Inf") and dist[s] + w < dist[d]:
                    print("Graph contains negative weight cycle")
                    return

        # No negative weight cycle found!
        # Output the shortest distance to each node
        return dist

    # --------------------------------------------------------
    # --------------------------------------------------------
    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):
        # Mark the current node as visited.
        visited.add(v)
        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if i not in visited:
                self.topologicalSortUtil(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.append(v)

    # The function to do Topological Sort
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = set()
        stack = []

        # Call the recursive helper function to store Topological-
        # Sort starting from all vertices one by one
        for i in self.graph:
            if i not in visited:
                self.topologicalSortUtil(i, visited, stack)

        return stack[::-1]  # return list in reverse order

    # Function to print MST using Prim's algorithm
    def prim_mst(self):
        pq = []  # Priority queue (Heap) to store vertices that are being processed
        src = 0  # Taking vertex 0 as the source
        # Create a list for keys and initialize all keys as infinite (INF)
        key = [float('inf')] * self.V
        # To store the parent array which, in turn, stores MST
        parent = [-1] * self.V
        # To keep track of vertices included in MST
        in_mst = [False] * self.V
        # Insert source itself into the priority queue and initialize its key as 0
        heapq.heappush(pq, (0, src))
        key[src] = 0

        # Loop until the priority queue becomes empty
        while pq:
            # The first vertex in the pair is the minimum key vertex
            # Extract it from the priority queue
            # The vertex label is stored in the second of the pair
            u = heapq.heappop(pq)[1]
            # Different key values for the same vertex may exist in the priority queue.
            # The one with the least key value is always processed first.
            # Therefore, ignore the rest.
            if in_mst[u]:
                continue
            in_mst[u] = True  # Include the vertex in MST
            # Iterate through all adjacent vertices of a vertex
            for v, weight in self.graph[u]:
                # If v is not in MST and the weight of (u, v) is smaller than the current key of v
                if not in_mst[v] and key[v] > weight:
                    # Update the key of v
                    key[v] = weight
                    heapq.heappush(pq, (key[v], v))

                    parent[v] = u

        # Print edges of MST using the parent array
        for i in range(1, self.V):
            print(f"{parent[i]} - {i}")



# --------------------------------------------------------
# --------------------------------------------------------
# Quick Select Partition: consider the last element as pivot
# and moves all smaller element to left of it and greater elements to right
def partition(arr, l, r):
    x = arr[r]
    i = l
    for j in range(l, r):
        if arr[j] <= x:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    arr[i], arr[r] = arr[r], arr[i]
    # i will be the index of cut
    return i

# find both kth largest element in array (only distinct elements)
# or kth (first element is 1st, no 0th) element in the sorted array (including repeating elements)
def kthSmallest(arr, l, r, k):
    # if k is smaller than number of elements in array
    if (k > 0 and k <= r - l + 1):

        # Partition the array around last element and get position of pivot element in sorted array
        index = partition(arr, l, r)

        # if position is same as k
        if (index - l == k - 1):
            return arr[index]

        # If position is more, repeat for left subarray
        if (index - l > k - 1):
            return kthSmallest(arr, l, index - 1, k)

        # Else repeat for right subarray
        return kthSmallest(arr, index + 1, r, k - index + l - 1)
    print("Index out of bound")




arr = [10, 4, 5, 8, 6, 11, 26]
print(kthSmallest(arr, 0, len(arr) - 1, k=3))





g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
# g.addEdge(0, 2, weight=2) #if weighted

visited = set()
g.DFSUtil(0, visited)