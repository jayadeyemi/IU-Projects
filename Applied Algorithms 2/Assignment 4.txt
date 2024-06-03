from math import sqrt

def helpRyan(points, sniperPoint):
    # Initialize an empty list to store enemy data
    enemy_data = []
    
    # Iterate through each enemy point
    for enemy in points:
        # Unpack coordinates
        x1, y1 = enemy[:2]
        x2, y2 = sniperPoint
        
        # Calculate the distance between the enemy and the sniper point
        distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        
        # Calculate the enemy's ammo supply
        ammo = sqrt(enemy[2])
        
        # Append enemy data (point, distance, ammo) to the list
        enemy_data.append([enemy, distance, ammo])

    # Sort the list based on the sum of distance and ammo using merge sort
    sorted_enemy_data = moded_mergeSort(enemy_data, key=lambda x: x[1] + x[2])

    # Return a list of enemy points sorted by distance and ammo
    return [enemy[0] for enemy in sorted_enemy_data]


def moded_mergeSort(data, key=lambda x: x): 
    def merge(left, right, key):
        # Initialize an empty list to store the merged result
        merged = []
        i = 0
        j = 0
        # Merge elements from left and right lists into the merged list
        while i < len(left) and j < len(right):
            if key(left[i]) <= key(right[j]):
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        # Append any remaining elements from left or right lists
        merged.extend(left[i:])
        merged.extend(right[j:])

        return merged
    # Base case: if the length of the data is 1 or less, return data
    if len(data) <= 1:
        return data
        
    # Recursively split the data into halves and sort each half
    left_half = moded_mergeSort(data[:len(data)//2], key)
    right_half = moded_mergeSort(data[len(data)//2:], key)

    # Merge the sorted halves
    return merge(left_half, right_half, key)





def jacksMachine(objects,query):
    n = len(objects)
    if query < 1 or query > n:
        return -1
    
    def quicksort(arr, low, high, k):

        def split_recursive_sort(arr, low, high):
            pivot = arr[high]  # Choose the last element
            i = low - 1  # Index of smaller element
            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]  # Swap elements
            arr[i + 1], arr[high] = arr[high], arr[i + 1]  # Swap pivot with element at i+1
            return i + 1
        
        if low < high:  
            pi = split_recursive_sort(arr, low, high) # split the list and find the kth item in the appriate split
            if pi == k:
                return arr[pi]  # if the index of the last item in the sublist matches, return the item
            elif pi < k:
                return quicksort(arr, pi + 1, high, k)  # else if the index of the last item in the sublist is less than the kth item, quicksort the second half
            else:
                return quicksort(arr, low, pi - 1, k)   # else if the index of the last item in the sublist is greater than the kth item, quicksort the second half
        else:
            return arr[low]  # else the last item in the main list is the k-th element
        
    return quicksort(objects, 0, n - 1, n - query)  # perform quicksort

def subarrayWithMinimumCost(harvestArray):
    def heapify(arr, n, i):
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[smallest]:
            smallest = left

        if right < n and arr[right] > arr[smallest]:
            smallest = right

        if smallest != i:
            arr[i], arr[smallest] = arr[smallest], arr[i]
            heapify(arr, n, smallest)
    
    firstitem = harvestArray[0]
    harvestArray.pop(0)
    
    length = len(harvestArray)

    # Build a min heap.
    for i in range(length // 2 - 1, -1, -1):
        heapify(harvestArray, length, i)

    # Extract elements from the heap in decreasing order.
    for i in range(length - 1, 0, -1):
        harvestArray[0], harvestArray[i] = harvestArray[i], harvestArray[0]
        heapify(harvestArray, i, 0)
    
    return firstitem + min(sum(harvestArray[:2]), sum(harvestArray[1:3]), sum([harvestArray[0], harvestArray[2]]))

def maxEarnings(start_time, end_time, profit):

    def moded_mergeSort(events): 
        if len(events) <= 1:
            return events
        mid = len(events) // 2

        def merge(left, right):
            result = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i][0] <= right[j][0]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            result += left[i:]
            result += right[j:]
            return result
        
        left = moded_mergeSort(events[:mid])
        right = moded_mergeSort(events[mid:])
        return merge(left, right)
    
    # Combine start time, end time, and profit into tuples
    events = list(zip(start_time, end_time, profit))
    n = len(events)
    events = moded_mergeSort(events)


    # Initialize an array to store maximum earnings at each event
    dp_profit = [-1] * n
    
    for i in range(n):
        dp_profit[i] = events[i][2]
        for j in range(i):
            if events[j][1] <= events[i][0]:
                dp_profit[i] = max(dp_profit[i], dp_profit[j] + events[i][2])



    # Return the maximum profit among all events
    return max(dp_profit)

def minimumCost(blossomCosts):
    n = len(blossomCosts)
    tab = [float('inf')] * (n + 1)
    tab[0] = 0

    for i in range(n):
        for j in range(i + 1, min(2 * i + 2, n) + 1):
            tab[j] = min(tab[j], tab[i] + blossomCosts[i])

    return tab[n]


def maxCoins(nums):
    # Add 1 to the beginning and end of the nums array
    nums = [1] + nums + [1]
    n = len(nums)
    
    # Create a DP table to store the maximum coins for each subproblem
    dp = [[0] * n for _ in range(n)]
    
    # Iterate over all possible balloon ranges
    for length in range(2, n):
        for left in range(0, n - length):
            right = left + length
            # Try each balloon as the last one to burst in the range
            for i in range(left + 1, right):
                dp[left][right] = max(dp[left][right], 
                                      nums[left] * nums[i] * nums[right] + dp[left][i] + dp[i][right])
    
    # Return the maximum coins for the entire range
    return dp[0][n - 1]



import heapq

class Huffman:
    def __init__(self):
        self.huffman_codes = {}
        self.source_string = ""

    def set_source_string(self, src_str):
        self.source_string = src_str

    def generate_codes(self):
        frequency = {}
        for char in self.source_string:
            frequency[char] = frequency.get(char, 0) + 1

        heap = [[weight, [char, ""]] for char, weight in frequency.items()]
        heapq.heapify(heap)  # Use heapq.heapify for proper heap creation

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            for pair in left[1:]:
                pair[1] = '0' + pair[1]
            for pair in right[1:]:
                pair[1] = '1' + pair[1]

            heapq.heappush(heap, [left[0] + right[0]] + left[1:] + right[1:])

        self.huffman_codes = dict(sorted(heap[0][1:], key=lambda p: (len(p[-1]), p)))
        return heap  # Return the generated Huffman tree

    def encode_message(self, message_to_encode):
        encoded_msg = ""
        for char in message_to_encode:
            encoded_msg += self.huffman_codes[char]
        return encoded_msg

    def decode_message(self, encoded_msg):
        decoded_msg = ""
        current_code = ""

        for bit in encoded_msg:
            current_code += bit
            for char, code in self.huffman_codes.items():
                if code == current_code:
                    decoded_msg += char
                    current_code = ""
                    break

        return decoded_msg
    
class Node:
   def __init__(self, stored_value, left_child=None, right_child=None):
       self.stored_value = stored_value
       self.left_child = left_child
       self.right_child = right_child

class Wavelet_Tree:
    def __init__(self, input_array):
        self.root_node = self.construct_wavelet_tree(input_array, 0, 9)

    def construct_wavelet_tree(self, input_array, lower_bound, upper_bound):
        if lower_bound == upper_bound:
            return Node('X' * len(input_array))

        midpoint = (lower_bound + upper_bound) // 2
        partitioning_bits = []
        for element_index in range(len(input_array)):
            if input_array[element_index] > midpoint:
                partitioning_bits.append(1)
            else:
                partitioning_bits.append(0)
        partitioning_string = "".join(map(str, partitioning_bits))
        current_node = Node(partitioning_string)

        left_subtree_array = []
        right_subtree_array = []
        for element in input_array:
            if element <= midpoint:
                left_subtree_array.append(element)
            else:
                right_subtree_array.append(element)

        current_node.left_child = self.construct_wavelet_tree(left_subtree_array, lower_bound, midpoint)
        current_node.right_child = self.construct_wavelet_tree(right_subtree_array, midpoint + 1, upper_bound)

        return current_node
    
    def get_wavelet_level_order(self):
        result = []
        queue = [self.root]

        while queue:
            level_nodes = queue  # Store nodes for the current level
            queue = []
            for node in level_nodes:
                if node:
                    result.append(node.value)
                    queue.append(node.left)
                    queue.append(node.right)

        return result

    def rank(self, target_character, position_index):
        return self.recursive_rank_helper(self.root_node, 0, 9, target_character, position_index)

    def recursive_rank_helper(self, current_node, lower_bound, upper_bound, target_character, position_index):
        if lower_bound == upper_bound:
            return position_index

        if not current_node or position_index == 0:
            return 0

        midpoint = (lower_bound + upper_bound) // 2
        count_of_zeros = 0
        substring = current_node.stored_value[:position_index]
        for character in substring:
            if character == '0':
                count_of_zeros += 1
        count_of_ones = position_index - count_of_zeros

        if target_character <= midpoint:
            return self.recursive_rank_helper(current_node.left_child, lower_bound, midpoint, target_character, count_of_zeros)
        else:
            return self.recursive_rank_helper(current_node.right_child, midpoint + 1, upper_bound, target_character, count_of_ones)
