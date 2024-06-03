import random

# Node class for SkipList
class Node:
    # initialize the Node
    def __init__(self, val):
        self.val = val  # value of the node
        self.next = None    # next pointer
        self.down = None    # down pointer

# SkipList class
class SkipList:
    # initialize the SkipList
    def __init__(self):
        self.head = Node(float("-inf")) # head of the SkipList
        self.levels = 1 # number of levels in the SkipList
        self.size = 0   # number of nodes in the SkipList
    # search for a value in the SkipList
    def search(self, target: int) -> bool:
        current = self.head # start from the head
        path = []   # path to the target
        while current:  # traverse the SkipList
            while current.next and current.next.val < target:   # for the current level, until the next value is greater than the target
                current = current.next  # move to the right
                path = path + [current.val]   # add the value to the path
            if current.down:    # once the current level is done
                current = current.down  # move down
                path = path + [current.val]  # add the value to the path
            else:   # if there is no lower level
                break  # break the loop
        if current.val == target or (current.next and current.next.val == target):  # if the target is found
            print("Found",path) # print the path
            return True # return True
        return False    # else return False
            
    # insert a value in the SkipList
    def insert(self, num: int) -> None:
        current = self.head # start from the head
        # length by level of the SkipList
        tower = [self.head] # tower of nodes
        path = []   # path to the target
        
        while current:  # traverse the SkipList
            while current.next and current.next.val < num:  # for the current level, until the next value is greater than the target
                current = current.next  # move to the right
                path = path + [current.val]  # add the value to the path
            if current.down:    # once the current level is done
                current = current.down  # move down
                path = path + [current.val] # add the value to the path
            else:   # if there is no lower level
                break   # break the loop
            tower.append(current)
        
        # add the new node to the SkipList
        new_node = Node(num)    # create a new node
        while tower:
            node = tower.pop()
            new_node.next = node.next
            node.next = new_node
            new_node = Node(num)
            new_node.down = node
        
        # Increase the number of levels probabilistically
        while random.random() < 0.5:
            new_level = Node(float("-inf"))  # Create a new level
            new_level.down = self.head  # Point it down to the level below
            self.head = new_level  # Update the head
            tower.append(new_level)  # Add the new level to the tower
            self.levels += 1  # Increase the number of levels
        
        self.size += 1  # Increase the size of the SkipList

                    
                
        

        # local update[0...MaxLevel+1]
        # x := list -> header
        # for i := list -> level downto 0 do
        #     while x -> forward[i] -> key  forward[i]
        # update[i] := x
        # x := x -> forward[0]
        # lvl := randomLevel()
        # if lvl > list -> level then
        # for i := list -> level + 1 to lvl do
        #     update[i] := list -> header
        #     list -> level := lvl
        # x := makeNode(lvl, searchKey, value)
        # for i := 0 to level do
        #     x -> forward[i] := update[i] -> forward[i]
        #     update[i] -> forward[i] := x



                

            
#sample test cases
# search 74 and insert 1 to 100
sl = SkipList()
for i in range(1, 100):
    sl.insert(i)
print(sl.search(74)) 
        


# example 1:
sl = SkipList()
for i in range(1, 100):
    sl.insert(i)
print(sl.search(50))
print(sl.search(74))
print(sl.search(100))
            

class amor_dict():
    def __init__(self, num_list = []):
        # your code here
        pass
    def insert(self, num):
        # your code here
        pass
    def search(self, num):
        # your code here
        pass
    def print(self):
        # Sample print function
        result = list()
        for level in self.levels: # iterate over all the levels
            result.append(level[:]) # make a copy of each level to result
        return result
    
class Deque:

    # Initializes the object with the size of the deque to be 'max_size'.
    def __init__(self, max_size: int):
        pass

    # Inserts an element at the front of deque;
    # Return True if the operation is successful, else False.
    def pushFront(self, value: int) -> bool:
        pass

    # Inserts an element at the rear of deque;
    # Return True if the operation is successful, else False.
    def pushRear(self, value: int) -> bool:
        pass

    # Return the front value from the deque;
    # if the deque is empty, return -1.
    def popFront(self) -> int:
        pass

    # Return the rear value from the deque;
    # if the deque is empty, return -1.
    def popRear(self) -> int:
        pass