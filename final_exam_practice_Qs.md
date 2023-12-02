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
