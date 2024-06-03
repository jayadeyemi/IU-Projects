
def bollywoodCharm(scripts,expectedCharm):
    current_min = 0
    current_max = len(scripts)-1
    while current_min < current_max:
        currrent_charm = scripts[current_min] + scripts[current_max]
        if expectedCharm == currrent_charm:
            return [current_min, current_max]
        elif expectedCharm > currrent_charm:
            current_min += 1
        else:
            current_max -= 1

def shiftingCharacters(inputStr, moves) -> str:
    # convert string to list
    inputStr = [i for i in inputStr]                          
    # create a dictionary to find letter equivalents of numbers
    find_number = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 
                   'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 
                   'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 
                   'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
    # create a dictionary to find number equivalents of letters     
    find_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 
                   7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 
                   14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 
                   20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}

    # in length of string is 1
    if len(inputStr) == 1:
        inputStr[0] = (find_number[inputStr[0]] + moves[0]) % 26       # convert string to number and add moves
        inputStr[0] = find_letter[inputStr[0]]                              # convert number to back to string
    else:
        # else iterate for length of the input string from back of the list
        for i in range(len(inputStr)):
            position = len(inputStr) - i - 1                                        # mark position
            moves[position-1] += moves[position]                                        # increment number of moves for the backward position
            inputStr[position] = (find_number[inputStr[position]] + moves[position]) % 26   # convert string to number and add moves
            inputStr[position] = find_letter[inputStr[position]]                                # convert number to back to string

        inputStr = ''.join(inputStr)
    return inputStr

def checkIsGeneDerived(G1,G2) -> bool:
    isVariant = False           # initialize the return value

    # Check if the first string is longer than the second
    if len(G1) > len(G2):       # if the first string is longer than the second, return false
        return isVariant

    # create a dictionary of the n_count of each character in the first string
    G1_dict = {}
    for i in G1:                # iterate through the first string
        if i in G1_dict:                    # if the character is in the dictionary, increment the n_count
            G1_dict[i] += 1
        else:                               # else, add the character to the dictionary
            G1_dict[i] = 1


    # create a similar dictionary for the second string in a window of the first string size
    G2_dict = {}
    for i in range(len(G1)):    # iterate through the second string using a window the size of the first string
        if G2[i] in G2_dict:                # if the character is in the dictionary, increment the n_count
            G2_dict[G2[i]] += 1
        else:                               # else, add the character to the dictionary
            G2_dict[G2[i]] = 1
   
    # check if the dictionaries are equal
    if G1_dict == G2_dict:      # if the dictionaries are equal, return true
        isVariant = True
    else:                       # else shift the window and check again
        for i in range(len(G1), len(G2)):   # iterate the window shift through the second string

            G2_dict[G2[i - len(G1)]] -= 1           # remove the character exiting the window from the dictionary
            if G2_dict[G2[i - len(G1)]] == 0:       # delete the character from the dictionary if the n_count is zero
                G2_dict.pop(G2[i - len(G1)])
            if G2[i] in G2_dict:                    # if the new character entering the window is in the dictionary, increment the n_count
                G2_dict[G2[i]] += 1             
            else:                                       # else, add the character to the dictionary
                G2_dict[G2[i]] = 1
            if G1_dict == G2_dict:                  # if the dictionaries are equal, mark presence as true and break out of the loop
                isVariant = True
                break
    return isVariant

def cheapestWarhead(valuations: list[int])->int:
    current_min = valuations[0]
    maxdamage = 0
    for i in range(len(valuations)):
        current_min = min(valuations[i], current_min)
        maxdamage = max(maxdamage, valuations[i] - current_min)

    return maxdamage

def specialString(n) -> int:
    # initialize base pattern and other parameters  
    pattern = [1, 2, 2]             # initial pattern
    pos = 1                         # is the postion that references the count of additions to be made
    n_count = 1                     # n_count is the number of 1s in the list
    last_item = 2                   # last_item is the last item on the list
       
    # keep shifting the pos to the right until the list length is "n" or "n + 1"
    while len(pattern) < n:
        pos += 1 
        if pattern[pos] == 1 and last_item == 1:        # if the last item on the list is 1 and the item in the reference position is 1 
                pattern.append(2)                           # append a quantity of one 2s to the list
                last_item = 2                                   # update the last item to be 2
        elif pattern[pos] == 1 and last_item == 2:      # if the last item on the list is 2 and the item in the reference position is 1 
                pattern.append(1)                           # append a quantity of one 1s to the list
                last_item = 1                                   # update the last item to be 1
                n_count +=1                                         # increment the count of 1s in the list by 1 
        elif pattern[pos] == 2 and last_item == 1:      # if the last item on the list is 1 and the item in the reference position is 2 
                pattern.extend([2, 2])                      # change the new last item to 2 and append a quantity of 1
                last_item = 2                                   # update the last item to be 2
        else:                                           # if the last item on the list is 2 and the item in the reference position is 2 
                pattern.extend([1, 1])                      # change the new last item to 2 and append a quantity of 1
                last_item = 1                                   # update the last item to be 2
                n_count +=2                                         # increment the count of 1s in the list by 1 

    # After the loop is completed check if last addition to the list was two 1s:
    if len(pattern)> n and pattern[-2] == pattern[-1] == 1:     # if it is two 1s, and the number of items in the list exceeds n
        n_count -=1                                                 # do not count the last item (1)

    return n_count  # return the count of 1s

def maxWaterStorage(pillarsHeight: list[int]) -> int:
    forwardmax = 0
    bckwardmax = len(pillarsHeight) - 1
    maxarea = 0
    
    # loop from two ends of the list to update the maximum area in the list.
    while forwardmax < bckwardmax:
        maxarea=max(maxarea, min(pillarsHeight[forwardmax], 
                                 pillarsHeight[bckwardmax]) * (bckwardmax- forwardmax))   # maximum_area = max(maximum_area.self, minimum_height * distance_between_pointers)
        
        # check to alternate between backward and forward pointer shift
        if pillarsHeight[forwardmax]>pillarsHeight[bckwardmax]:
            bckwardmax -=1  # shift right pointer backward when left pointer is larger
        else:
            forwardmax +=1  # shift left pointer forward when right pointer is larger
    
    # after loop ends, return the maximum area
    return maxarea

def cycle_buster(node):
    # Initialize two pointers, slow and fast
    slow = node
    fast = node

    # iterate through the linked list
    while fast is not None and fast.next is not None:
        # defining two pointers to detect loop
        slow = slow.next        # Move slow pointer one step
        fast = fast.next.next   # Move fast pointer two step

        # If there is a cycle, find the start of the cycle
        if slow == fast:
            slow = node             # restart the slow counter at beginning of linked list
            prev = fast             # initialise a variable for the previous position of the fast pointer
            
            # keep moving the pointers until they meet
            while slow != fast:
                prev = fast             # update the previous pointer position
                slow = slow.next        # Move slow pointer one step
                fast = fast.next        # Move fast pointer one step

            prev.next = None        # previous pointer is just before the loop, so set next to none to break cycle
            return node

    # If no cycle is detected, return head
    return [node]

import os
import glob
import nbformat as nbf

def py_to_ipynb(py_file_path):
    with open(py_file_path, 'r') as py_file:
        py_code = py_file.read()

    notebook = nbf.v4.new_notebook()
    notebook.cells.append(nbf.v4.new_code_cell(py_code))

    ipynb_file_path = py_file_path.replace('.py', '.ipynb')
    with open(ipynb_file_path, 'w') as ipynb_file:
        nbf.write(notebook, ipynb_file)

    print(f"Converted {py_file_path} to {ipynb_file_path}")

def convert_all_py_to_ipynb(folder_path):
    py_files = glob.glob(os.path.join(folder_path, '*.py'))
    for py_file in py_files:
        py_to_ipynb(py_file)

# Specify the folder path containing the .py files
folder_path = r'C:\Users\babas\OneDrive - Indiana University\Documents\Masters-Career Documents\Portfolio\IU-Works\Applied Algorithms 2'
convert_all_py_to_ipynb(folder_path)

