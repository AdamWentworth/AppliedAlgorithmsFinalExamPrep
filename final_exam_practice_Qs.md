# Question 1: Data Structures and Recursion

Problem Statement: Implement a function that uses recursion to reverse a linked list.

Solution:
```
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def reverse_linked_list(head):
    if not head or not head.next:
        return head
    new_head = reverse_linked_list(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

Example usage
```
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
reversed_head = reverse_linked_list(head)
```

# Question 2: Sorting Algorithms - Mergesort and Quicksort

Problem Statement: Implement both Mergesort and Quicksort algorithms. Compare their efficiencies on the same random list of numbers.

Solution:

```
import random

def mergesort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        mergesort(L)
        mergesort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```
Example usage
```
arr = [random.randint(0, 100) for _ in range(100)]
print("Original Array:", arr)
print("Mergesort:", mergesort(arr.copy()))
print("Quicksort:", quicksort(arr.copy()))
```

# Question 3: Trees and Recursion

Problem Statement: Write a function that uses recursion to calculate the height of a binary tree. The height of a tree is the number of nodes along the longest path from the root node down to the farthest leaf node.

Solution:

```
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def tree_height(node):
    if not node:
        return 0
    else:
        left_height = tree_height(node.left)
        right_height = tree_height(node.right)
        return max(left_height, right_height) + 1
```
Example usage
```
root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
print(tree_height(root))  # Should return the height of the tree
```

# Question 4: Benchmarking and Big O
Problem Statement: Implement a function that compares the time complexity of two sorting algorithms: bubble sort and insertion sort. Use this to empirically demonstrate their Big O notation on lists of varying sizes.

Solution:
```
import time
import random

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >=0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key

def benchmark(func, arr):
    start_time = time.time()
    func(arr.copy())
    return time.time() - start_time

# Generate lists of varying sizes
list_sizes = [100, 200, 500, 1000]
for size in list_sizes:
    test_arr = [random.randint(0, 1000) for _ in range(size)]
    print(f"Size: {size}, Bubble Sort Time: {benchmark(bubble_sort, test_arr)}")
    print(f"Size: {size}, Insertion Sort Time: {benchmark(insertion_sort, test_arr)}")
```

# Question 5: Stacks, DFS, and Recursion

Problem Statement: Write a function that uses a stack (without recursion) to perform a depth-first search on a binary tree. The function should return the list of values in the order they were visited.

Solution:

```
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def depth_first_search(root):
    if not root:
        return []
    stack, result = [root], []
    while stack:
        node = stack.pop()
        result.append(node.value)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result
```
Example usage
```
root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
print(depth_first_search(root))  # Should return [1, 2, 4, 5, 3]
```
# Question 6: Sorting and Asymptotics
Problem Statement: Implement the selection sort algorithm in Python. Then analyze its time complexity in terms of Big O notation.

Solution:

```
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr
```
Example usage
```
arr = [64, 25, 12, 22, 11]
sorted_arr = selection_sort(arr)
print(sorted_arr)  # Sorted array
```
Time Complexity Analysis: O(n^2) - as there are two nested loops.

# Question 7: Queues, BFS, and Trees

Problem Statement: Implement a function to perform a level-order traversal (breadth-first search) on a binary tree using a queue. The function should return a list of values in level-order.

Solution:
```
from collections import deque

class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def level_order_traversal(root):
    if not root:
        return []
    queue = deque([root])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result
```
Example usage
```
root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
print(level_order_traversal(root))  # Should return [1, 2, 3, 4, 5]
```
# Question 8: Data Structures - Stacks and Queues

Problem Statement: Implement a Python class that simulates a stack using two queues. The class should support push, pop, and top operations.

Solution:
```
from collections import deque

class StackUsingQueues:
    def __init__(self):
        self.queue1 = deque()
        self.queue2 = deque()

    def push(self, x):
        self.queue2.append(x)
        while self.queue1:
            self.queue2.append(self.queue1.popleft())
        self.queue1, self.queue2 = self.queue2, self.queue1

    def pop(self):
        return self.queue1.popleft()

    def top(self):
        return self.queue1[0]
```
Example usage
```
stack = StackUsingQueues()
stack.push(1)
stack.push(2)
print(stack.top())  # Returns 2
stack.pop()
print(stack.top())  # Returns 1
```

# Question 9: Sorting - Quicksort

Problem Statement: Modify the quicksort algorithm to sort a list of strings based on their length. Analyze its time complexity.

Solution:

```
def quicksort_strings(arr):
    if len(arr) <= 1:
        return arr
    pivot = len(arr[len(arr) // 2])
    left = [x for x in arr if len(x) < pivot]
    middle = [x for x in arr if len(x) == pivot]
    right = [x for x in arr if len(x) > pivot]
    return quicksort_strings(left) + middle + quicksort_strings(right)
```
Example usage
```
arr = ["apple", "banana", "cherry", "date"]
sorted_arr = quicksort_strings(arr)
print(sorted_arr)  # Sorted array based on string length
```
Time Complexity Analysis:
The time complexity remains O(n log n) on average, but it can degrade to O worst case

# Question 10: Linked Lists and Big O
Problem Statement: Implement a function to find the middle node of a singly linked list. If the list has an even number of nodes, return the second middle node. Analyze its time complexity in terms of Big O notation.

Solution:

```
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def find_middle_node(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```
Example usage
```
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
middle_node = find_middle_node(head)
print(middle_node.value)  # Should return 3
```
Time Complexity Analysis: O(n) - where n is the number of nodes in the list. The fast pointer traverses the list at twice the speed of the slow pointer, thus halving the time needed.

# Question 11: Recursion and Sorting
Problem Statement: Write a recursive function in Python that sorts an array using insertion sort.

Solution:

```
def insertion_sort_recursive(arr, n):
    if n <= 1:
        return

    insertion_sort_recursive(arr, n-1)
    last = arr[n-1]
    j = n-2
    
    while j >= 0 and arr[j] > last:
        arr[j + 1] = arr[j]
        j = j - 1

    arr[j + 1] = last
```
Example usage
```
arr = [12, 11, 13, 5, 6]
insertion_sort_recursive(arr, len(arr))
print("Sorted array is:", arr)
```
Time Complexity Analysis: O(n^2) - as each element is compared with all the other elements in a sorted sub-array.

# Question 12: Benchmarking Data Structures
Problem Statement: Compare the performance of a Python list (dynamic array) and a deque (double-ended queue) from the collections module for inserting elements at the beginning.

Solution:

```
import time
import random
from collections import deque

def benchmark_list_insertion(n):
    start_time = time.time()
    lst = []
    for _ in range(n):
        lst.insert(0, random.randint(0, 100))
    return time.time() - start_time

def benchmark_deque_insertion(n):
    start_time = time.time()
    dq = deque()
    for _ in range(n):
        dq.appendleft(random.randint(0, 100))
    return time.time() - start_time

n = 10000
print("List Insertion Time:", benchmark_list_insertion(n))
print("Deque Insertion Time:", benchmark_deque_insertion(n))
```
Time Complexity Analysis: Insertion at the beginning is O(n) for a list and O(1) for a deque.

# Question 13: Trees and BFS
Problem Statement: Implement a function that checks if a binary tree is a complete binary tree using breadth-first search (BFS).

Solution:
```
from collections import deque

class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def is_complete_binary_tree(root):
    if not root:
        return True
    queue = deque([root])
    end = False
    while queue:
        node = queue.popleft()
        if node:
            if end:
                return False
            queue.append(node.left)
            queue.append(node.right)
        else:
            end = True
    return True
```
Example usage
```
root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3, None, TreeNode(6)))
print(is_complete_binary_tree(root))  # Should return False
```
# Question 14: Asymptotics and Linked Lists
Problem Statement: Write a function to merge two sorted linked lists into a single sorted linked list. Analyze its time complexity in terms of Big O notation.

Solution:
```
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def merge_two_sorted_lists(l1, l2):
    dummy = ListNode()
    tail = dummy
    while l1 and l2:
        if l1.value < l2.value:
            tail.next, l1 = l1, l1.next
        else:
            tail.next, l2 = l2, l2.next
        tail = tail.next
    tail.next = l1 or l2
    return dummy.next
```
Example usage
```
list1 = ListNode(1, ListNode(3, ListNode(5)))
list2 = ListNode(2, ListNode(4, ListNode(6)))
merged_list = merge_two_sorted_lists(list1, list2)
```
Time Complexity Analysis: O(n + m) - where n and m are the lengths of the two lists.

# Question 15: Recursion in Data Structures
Problem Statement: Write a recursive function to count the number of nodes in a singly linked list.

Solution:
```
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def count_nodes_recursive(head):
    if not head:
        return 0
    return 1 + count_nodes_recursive(head.next)
```
Example usage
```
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
print(count_nodes_recursive(head))  # Should return 4
```
# Question 16: Benchmarking Sorting Algorithms
Problem Statement: Benchmark the performance of insertion sort and selection sort on randomly generated lists of various sizes. Compare their execution times.

Solution:
```
import time
import random

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >=0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

def benchmark(func, arr):
    start_time = time.time()
    func(arr.copy())
    return time.time() - start_time

# Generate lists of varying sizes
list_sizes = [100, 200, 500, 1000]
for size in list_sizes:
    test_arr = [random.randint(0, 1000) for _ in range(size)]
    print(f"Size: {size}, {func.__name__} Time: {benchmark(func, test_arr)}")
```
# Question 17: Recursion and Trees
Problem Statement: Write a function that uses recursion to find the maximum value in a binary tree.

Solution:
```
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def find_max_in_tree(node):
    if not node:
        return float('-inf')
    max_left = find_max_in_tree(node.left)
    max_right = find_max_in_tree(node.right)
    return max(node.value, max_left, max_right)
```
Example usage
```
root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
print(find_max_in_tree(root))  # Should return 5
```
# Question 18: Linked Lists and Iteration
Problem Statement: Write an iterative function to reverse a singly linked list.

Solution:
```
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def reverse_linked_list_iterative(head):
    prev, curr = None, head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev
```
Example usage
```
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
reversed_head = reverse_linked_list_iterative(head)
```
# Question 19: Sorting and Benchmarking
Problem Statement: Implement bubble sort and analyze its time complexity. Then, benchmark its performance on a large dataset.

Solution:
```
import time
import random

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def benchmark(func, arr):
    start_time = time.time()
    func(arr.copy())
    return time.time() - start_time
```
Example usage
```
arr = [random.randint(0, 100) for _ in range(1000)]
print("Bubble Sort Time:", benchmark(bubble_sort, arr))
```
Time Complexity Analysis: O(n^2) - as it involves two nested loops.

# Question 20: Data Structures - Implementing a Queue using Stacks
Problem Statement: Implement a queue using two stacks in Python, with enqueue and dequeue operations.

Solution:
```
class QueueUsingStacks:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def enqueue(self, x):
        self.in_stack.append(x)

    def dequeue(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack.pop()
```
Example usage
```
queue = QueueUsingStacks()
queue.enqueue(1)
queue.enqueue(2)
print(queue.dequeue())  # Returns 1
queue.enqueue(3)
print(queue.dequeue())  # Returns 2
```
# Question 21: Recursion in Sorting
Problem Statement: Write a recursive function to perform selection sort.

Solution:
```
def selection_sort_recursive(arr, n, index = 0):
    if index == n:
        return
    
    min_index = index
    for j in range(index + 1, n):
        if arr[j] < arr[min_index]:
            min_index = j

    arr[index], arr[min_index] = arr[min_index], arr[index]
    selection_sort_recursive(arr, n, index + 1)
```
Example usage
```
arr = [64, 25, 12, 22, 11]
selection_sort_recursive(arr, len(arr))
print("Sorted array:", arr)
```
# Question 22: Trees - Finding a Node
Problem Statement: Write a function that searches for a given value in a binary search tree. Return true if the value exists in the tree, false otherwise.

Solution:
```
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def search_bst(root, val):
    if not root:
        return False
    if root.value == val:
        return True
    elif val < root.value:
        return search_bst(root.left, val)
    else:
        return search_bst(root.right, val)
```
Example usage
```
root = TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)), TreeNode(6, TreeNode(5), TreeNode(7)))
print(search_bst(root, 3))  # Should return True
print(search_bst(root, 8))  # Should return False
```
# Question 23: Asymptotics and Linked Lists
Problem Statement: Write a function to find the kth to last element of a singly linked list. Discuss the time complexity of your solution.

Solution:
```
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def find_kth_to_last(head, k):
    fast = slow = head
    for _ in range(k):
        if not fast:
            return None
        fast = fast.next
    while fast:
        slow = slow.next
        fast = fast.next
    return slow
```
Example usage
```
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
kth_node = find_kth_to_last(head, 2)
print(kth_node.value)  # Should return 4
```
Time Complexity Analysis: O(n) - where n is the number of elements in the list.
# Question 24: Sorting and Data Structures
Problem Statement: Implement a function that sorts a list of integers using a stack. Describe its time complexity.

Solution:
```
def sort_stack(arr):
    stack = []
    while arr:
        temp = arr.pop()
        while stack and stack[-1] > temp:
            arr.append(stack.pop())
        stack.append(temp)
    while stack:
        arr.append(stack.pop())
    return arr
```
Example usage
```
arr = [34, 3, 31, 98, 92, 23]
sorted_arr = sort_stack(arr)
print("Sorted array:", sorted_arr)
```
Time Complexity Analysis: O(n^2) - as each element is compared with others in a nested manner.

# Question 25: Trees - Depth of a Node
Problem Statement: Write a function to find the depth of a given node in a binary tree. If the node doesn't exist, return -1.

Solution:
```
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def find_depth(root, val, depth=0):
    if not root:
        return -1
    if root.value == val:
        return depth
    left_depth = find_depth(root.left, val, depth + 1)
    if left_depth != -1:
        return left_depth
    return find_depth(root.right, val, depth + 1)
```
Example usage
```
root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
print(find_depth(root, 5))  # Should return 2
print(find_depth(root, 8))  # Should return -1
```
# Question 26: Recursion in Trees
Problem Statement: Write a recursive function to count the number of leaf nodes in a binary tree.

Solution:
```
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def count_leaf_nodes(node):
    if not node:
        return 0
    if not node.left and not node.right:
        return 1
    return count_leaf_nodes(node.left) + count_leaf_nodes(node.right)
```
Example usage
```
root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
print(count_leaf_nodes(root))  # Should return 3
```
# Question 27: Linked Lists - Merge Two Sorted Lists
Problem Statement: Write a function that merges two sorted linked lists into a single sorted linked list. Discuss the time complexity of your solution.

Solution:
```
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

def merge_sorted_lists(l1, l2):
    dummy = ListNode()
    tail = dummy
    while l1 and l2:
        if l1.value < l2.value:
            tail.next, l1 = l1, l1.next
        else:
            tail.next, l2 = l2, l2.next
        tail = tail.next
    tail.next = l1 or l2
    return dummy.next
```
Example usage
```
list1 = ListNode(1, ListNode(3, ListNode(5)))
list2 = ListNode(2, ListNode(4, ListNode(6)))
merged_list_head = merge_sorted_lists(list1, list2)
```
Time Complexity Analysis: O(n + m) - where n and m are the lengths of the two lists.

# Question 28: Benchmarking Sorting Algorithms
Problem Statement: Compare the execution times of insertion sort and bubble sort on arrays of different sizes. Explain the observed differences in performance.

Solution:
```
import time
import random

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def benchmark_sorting_algorithm(algorithm, arr):
    start_time = time.time()
    algorithm(arr)
    return time.time() - start_time
```
Example usage
```
array_sizes = [100, 500, 1000]
for size in array_sizes:
    random_array = [random.randint(0, 1000) for _ in range(size)]
    time_taken = benchmark_sorting_algorithm(insertion_sort, random_array.copy())
    print(f"Insertion Sort on {size} elements: {time_taken} seconds")
    time_taken = benchmark_sorting_algorithm(bubble_sort, random_array.copy())
    print(f"Bubble Sort on {size} elements: {time_taken} seconds")
```
# Question 29: Trees - Binary Tree Maximum Path Sum
Problem Statement: Implement a function to find the maximum path sum in a binary tree. The path can start and end at any node in the tree.

Solution:
```
class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def max_path_sum(root):
    def helper(node):
        if not node:
            return 0
        left = max(helper(node.left), 0)
        right = max(helper(node.right), 0)
        return node.value + max(left, right)

    return helper(root)
```
Example usage
```
root = TreeNode(10, TreeNode(2, TreeNode(20), TreeNode(1)), TreeNode(10, None, TreeNode(-25, TreeNode(3), TreeNode(4))))
print(max_path_sum(root))  # Should return the maximum path sum
```
