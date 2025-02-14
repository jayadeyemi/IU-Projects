class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def allCombinationForces(n: int) -> list[str]:
    result = []
    aggregate = 0
    
    def generate_combinations(n_red, n_blue, combination, result, aggregate):
        # Base case: if both red and blue forces are exhausted, check if the combination is balanced
        # first and last character of the combination are R and B respectively
        if n_red == 0 and n_blue == 0:
            if is_balanced(combination):
                result.append(combination)
                
            return
        
        # Recursive case: try adding red force if available
        if n_red > 0:
            generate_combinations(n_red - 1, n_blue, combination + 'R', result, aggregate+1)
        
        # Recursive case: try adding blue force if available
        if n_blue > 0 and aggregate > 0:
            generate_combinations(n_red, n_blue - 1, combination + 'B', result, aggregate-1)

    def is_balanced(combination):
        count_red = combination.count('R')
        count_blue = combination.count('B')
        return count_red == count_blue and combination[0] == 'R' and combination[-1] == 'B'
    
    generate_combinations(n, n, '', result, aggregate)
    return result



def josephus_plot(n: int, k: int) -> int:
    soldiers = [i for i in range(1, n + 1)]

    def iterator(soldiers, k, index=0):
        if len(soldiers) == 1:
            return soldiers[0]

        index = (index + k - 1) % len(soldiers)
        del soldiers[index]
        return iterator(soldiers, k, index)
    
    return iterator(soldiers, k)


def shorterBuildings(heights):
    def merge_sort(arr, counts):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = merge_sort(arr[:mid], counts)
        right = merge_sort(arr[mid:], counts)
        return merge(left, right, counts)

    def merge(left, right, counts):
        merged = []
        i, j = 0, 0
        while i < len(left) and j < len(right):
            if left[i][1] > right[j][1]:
                merged.append(left[i])
                counts[left[i][0]] += len(right) - j
                i += 1
            else:
                merged.append(right[j])
                j += 1      
        merged.extend(left[i:])
        merged.extend(right[j:])
        print(merged)
        print(counts)
        return merged

    n = len(heights)
    counts = [0] * n
    print("count", counts)
    # Enumerate heights with indices
    enumerated_heights = list(enumerate(heights))
    merge_sort(enumerated_heights, counts)
    return counts


    
def printTree(root: TreeNode, level=0, prefix="Root: ") -> None:
    if root is not None:
        print(" " * (level * 4) + prefix + str(root.val))
    if root.left is not None:
        printTree(root.left, level + 1, "L: ")
    if root.right is not None:
        printTree(root.right, level + 1, "R: ")

def isHeightBalanced(root: TreeNode) -> bool:
    def checkBalance(node):
        if not node:
            return 0, True
        left_height, left_balanced = checkBalance(node.left)
        right_height, right_balanced = checkBalance(node.right)
        balanced = left_balanced and right_balanced and abs(left_height - right_height) <= 1
        return max(left_height, right_height) + 1, balanced
    
    _, balanced = checkBalance(root)
    return balanced       

def createBinaryTree(preorder: list[int], inorder: list[int]) -> TreeNode:
    if not preorder:
        return None
    root = TreeNode(preorder[0])
    root_index = inorder.index(preorder[0])
    root.left = createBinaryTree(preorder[1:root_index + 1], inorder[:root_index])
    root.right = createBinaryTree(preorder[root_index + 1:], inorder[root_index + 1:])
    return root        

def palindromic_paths(root):
    def dfs(node, path_counts):
        nonlocal count
        if not node:
            return

        # Update path counts with the current node's value
        path_counts[node.val] = path_counts.get(node.val, 0) + 1

        # If it's a leaf node, check if the path is palindromic
        if not node.left and not node.right:
            # Check if there's at most one value with an odd count
            if sum(count % 2 == 1 for count in path_counts.values()) <= 1:
                count += 1

        # Recursively visit left and right subtrees
        dfs(node.left, path_counts.copy())
        dfs(node.right, path_counts.copy())

    if not root:
        return 0

    count = 0
    dfs(root, {})
    return count


def minimumTime(root, reference_node):
    def find_path(root, node, path):
        if not root:
            return False
    
        # Add current node to the path
        path.append(root)

        # If the current node is the node we are looking for, return True
        if root.val == node:
            return True

        # If the node is found in the left or right subtree, return True
        if (root.left and find_path(root.left, node, path)) or (root.right and find_path(root.right, node, path)):
            return True
        
        # If the node is not found in this subtree, remove the current node from the path
        path.pop()
        return False

    def find_longest_path(node):
        nonlocal reversed_path
        if not node:
            return 0
        
        if node.left or node.right:
            # if reversed_path is ewmpty
            if reversed_path == [] or node.left != reversed_path[-1]:
                left = find_longest_path(node.left)
            else:
                left = 0
            if reversed_path == [] or node.right != reversed_path[-1]:
                right = find_longest_path(node.right)
            else:
                right = 0
        else:
            return 0
        
        return 1 + max(left, right)


    path_to_reference = []
    list_of_paths = []
    distance = 0
    node = None
    i=0
    reversed_path = []

    if not root:
        return None
    
    # Find the path from root to the reference node
    find_path(root, reference_node, path_to_reference)
    
    # Find the longest path from the reference node to a leaf node
    
    while path_to_reference:
        node = path_to_reference.pop()
        distance = max(distance, find_longest_path(node)+i)
        reversed_path.append(node)                             
        i+=1
    return distance


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

    def decode_message(self, encoded_msg, heap):  # Pass heap as an argument
        decoded_msg = ""
        current_code = ""
        current_node = heap[0]  # Access the final Huffman tree

        for bit in encoded_msg:
            current_code += bit
            for i in range(1, len(current_node)):
                char, code = current_node[i]
                if code == current_code:
                    decoded_msg += char
                    current_code = ""
                    current_node = heap[0]  # Reset to root for next character
                    break

        return decoded_msg
