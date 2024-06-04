import random
# Implementing a Skip List
# A skip list is a data structure that allows fast search within an ordered sequence of elements.

# Nodes class for the Skip List
class Node:
    def __init__(self, val):
        self.val = val  # the value of the node
        self.next = None    # the next node in the same level
        self.down = None    # the node in the below level

# Skip List class with search and insert methods
class SkipList:
    def __init__(self):
        self.head = Node(float("-inf"))
        self.levels = 0
        self.len = 0
        
    # search for a target number in the list
    def search(self, target: int) -> bool:
        current = self.head # start from the head
        path = []  # record the path of the search for debugging
        # progress to the bottom level of the list
        while current:
            path.append(current.val)
            while current.next and current.next.val < target:   # progress to the largest number on level that is less than the target
                current = current.next
            if current.next and current.next.val == target: # if the target is found, return True
                path.append(current.next.val)
                print(path, "Found")
                return True   
            current = current.down  # progress to the lower level if the target is not found
        print(path, "Not Found")
        return False    # At the bottom level, return False if the target is not found

    # insert a number into the list
    def insert(self, num: int) -> None:
        current = self.head # start from the head 
        tower = [] # tower retrieves the nodes on the path of the insertion for connecting the new node
        path = [self.head.val]  # record the path of the insertion for debugging
        # progress to the bottom level of the list
        while current:# progress to the bottom level of the list
            path.append(current.val)
            while current.next and current.next.val < num: # progress to the number before the target
                current = current.next
            tower.append(current)   # tower recordes the path of the current number in the bottom level
            path.append(current.val)    
            current = current.down  # progress to the lower level
            
        # initialize the new node and insert it into the bottom level
        node = Node(num)
        prev = tower.pop() # the number before the target
        node.next = prev.next   #  connect the new node to the next node
        prev.next = node    #  connect the previous node to the new node
        self.len += 1 # increase the size of the list
        level = 0 # the current level of the new node

        # Build the inserted node up to the higher levels with a coin flip
        while random.random() > 0.5:
            # extend the head if a new level is reached
            if level >= self.levels:
                # create a new head for the new level and connect it to the previous head as the new head
                new_head = Node(float("-inf"))
                new_head.down = self.head
                self.head = new_head
                # add the new head to the top of the tower and increase the number of levels
                tower.append(new_head)  
                self.levels += 1
            
            # connect the new node at the neighbors stored in the tower at the current level
            prev = tower.pop()  
            new_node = Node(num)
            new_node.next = prev.next
            prev.next = new_node
            new_node.down = node
            # convert the new node to the previous node for the next level
            node = new_node
            level += 1 
        # print(path, "Inserted")
            

class Deque:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.queue = []

    def pushFront(self, value: int) -> bool:
        if len(self.queue) < self.max_size:
            self.queue.insert(0, value)
            return True
        return False

    def pushRear(self, value: int) -> bool:
        if len(self.queue) < self.max_size:
            self.queue.append(value)
            return True
        return False

    def popFront(self) -> int:
        if self.queue:
            return self.queue.pop(0)
        return -1

    def popRear(self) -> int:
        if self.queue:
            return self.queue.pop()
        return -1

def saveMaximumPeople(cars: list[int]) -> int:
    sum_cars = 0
    last_item = 0
    while cars:
        current_item = cars.pop()  
        if current_item > 0 and last_item <= 0:
            if abs(current_item) > abs(last_item):
                sum_cars -= abs(last_item)
                sum_cars += max(current_item, abs(last_item))
                print("+ve, -ve", sum_cars, current_item, last_item)
        elif current_item > 0 and last_item >= 0:
            sum_cars += current_item
            print("+ve, +ve", sum_cars, current_item, last_item)
        elif current_item < 0 and last_item <= 0:
            sum_cars += max(abs(current_item), abs(last_item))
            print("-ve, -ve", sum_cars, current_item, last_item)
        elif current_item < 0 and last_item >= 0:
            sum_cars += abs(current_item)
            print("-ve, +ve", sum_cars, current_item, last_item)
        elif abs(current_item) == abs(last_item):
            sum_cars += 0
            print("==", sum_cars, current_item, last_item)
        last_item = current_item
    return sum_cars       
    


def saveMaximumPeople(cars):
    saved = []
    previous = cars.pop()
    saved.append(previous)
    while len(cars) > 0:
        current = cars.pop()
        
        if current == abs(previous):
            if saved:
                saved.pop()
            continue
        elif current > 0:
            if previous < 0:
                if saved:
                    saved.pop()
                saved.append(max(current, abs(previous)))
            else:
                saved.append(current)
        elif current < 0:
                saved.append(abs(current))
        
        print(saved)
        previous = current
    sum_saved = sum(saved)
    return sum_saved

# test cases
print(saveMaximumPeople([-3,-4, -5, 5,6,7])) # 30
print(saveMaximumPeople([1,-1,1,-1, 3, 3])) # 0
print(saveMaximumPeople([-2,-2,1,-2])) # 6
print(saveMaximumPeople([5,-5,-5,1, -1])) # 5
print(saveMaximumPeople([4,-2,10,6,9,-4]))

def ironman_vs_thanos_again(s):
    stack = []  # Initialize an empty stack to store 'A's
    back_stack = []  # Initialize an empty stack
    longest_balanced = 0  # Initialize the length of the longest balanced part
    max_balanced = 0  # Initialize the length of the maximum balanced part  
    last_item = None
    for char in s:
        if char == 'A':
            stack.append(char)  # Push 'A' onto the stack
            back_stack.append(char)
        elif char == 'T' and stack:  # If 'T' is encountered and stack is not empty
            back_stack.append(char)
            stack.pop()  # Pop matching 'A' from the stack
            longest_balanced += 2  # Increment the length of the longest balanced part
        else:
            longest_balanced = 0

        max_balanced = max(longest_balanced, max_balanced)
        print(stack, longest_balanced, max_balanced) 
    if stack and back_stack[-1]=='T':
        max_balanced -= 2
    return max_balanced

def impatientBob(Prices: list[int]) -> list[int]:
    n = len(Prices)
    stack = []
    result = [0] * n
    
    for i in range(n):
        while stack and Prices[i] > Prices[stack[-1]]:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)
    
    return result

def scoreOfMagicalString(s: str) -> int:
    stack = []
    cur_score = 0
    
    for char in s:
        if char == '(':
            stack.append(cur_score)
            cur_score = 0
        else:  # char == ')'
            cur_score = stack.pop() + max(cur_score * 2, 1)
    
    return cur_score

class amor_dict():
    def __init__(self, num_list=None):
        self.levels = []
        if num_list is not None:
            for num in num_list:
                self.insert(num)

    def insert(self, num):
        def find_exponents(level_count):
            result = []
            i = 0
            while level_count > 0:
                if level_count % 2 == 1:
                    result.append(i)
                level_count //= 2
                i += 1
            return result
        def merge_sort(lst):
            if len(lst) <= 1:
                return lst
            mid = len(lst) // 2
            left = merge_sort(lst[:mid])
            right = merge_sort(lst[mid:])
            return merge(left, right)
        
        def merge(left, right):
            merged = []
            left_index = right_index = 0
            while left_index < len(left) and right_index < len(right):
                if left[left_index] <= right[right_index]:
                    merged.append(left[left_index])
                    left_index += 1
                else:
                    merged.append(right[right_index])
                    right_index += 1
            merged.extend(left[left_index:])
            merged.extend(right[right_index:])
            return merged
        level_index = 0
        while level_index < len(self.levels) and len(self.levels[level_index]) == 2 ** level_index:
            level_index += 1

        if level_index == len(self.levels):
            self.levels.append([])

        self.levels[level_index].append(num)
        self.levels[level_index] = merge_sort(self.levels[level_index])

    def search(self, num):
        def binary_search(level, num):
                left = 0
                right = len(level) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if level[mid] == num:
                        return True
                    elif level[mid] < num:
                        left = mid + 1
                    else:
                        right = mid - 1
        level_index = -1
        for level in self.levels:
            level_index += 1
            if binary_search(level, num):
                return level_index
        return -1

    
    def print(self):
        result = []
        for level in self.levels:
            result.append(level[:])
        return str(result)


# Example usage:
dynamic_array = amor_dict([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
print(dynamic_array)  # Output: [[1], [2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
print(dynamic_array.search(10))  # Output: True
print(dynamic_array.search(16))  # Output: False

# test cases
# create a new amor_dict object
ad = amor_dict([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# print the levels of the amor_dict
print(ad.print()) 
