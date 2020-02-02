---
layout: post
title: "Amazon code interview review"
author: Guanlin Li
tag: blog
---

* [Day 1](#day-1)
* [Day 4](#day-4)
* [Day 5](#day-5)



The review philosophy is fast + slow review, just like fast/slow change of memory in brain:

> Two adjacent days for reviewing the same topics twice.

### Day 1

> Jan. 16-17

#### String related

- `atoi` implementation: `ord` is to look-up a char's ordinary number in the ASCII code
- `max_int = 2 ** 31 - 1`, and `min_in = - 2 ** 32`: 2147483647, and -2147483648
- Make sure the usage of variables are consistent.



### Day 4

> Jan. 24 (trip to CD, with virus spreadout)

#### Longest consecutive sequence

> Given an array, find the longest consecutive sequence's length.

Use `hash_map` to store the array's elements, with `O(1)` access time.

- The second solution is not understood, using `union` and `find` functions of the `map` class.

#### Two sum

> Given any unordered array, and a target sum, find the two indexes that represent the numbers' sum equaling to the target sum.

Use `hass_map` instead of sorting, to get `O(1)` complexity. `左右夹逼`

```python
# Given a target t and a sorted array nums, complexity is O(n)
res = []
i = 0
j = len(nums) - 1
while i < j:
  if nums[i] + nums[j] < t:
    i += 1
  elif nums[i] + nums[j] > t:
    j -= 1
  else:
    res.append((i, j))
```

- Related questions
  - Three sum, Three sum closest, 4sum (pre-calc two numbers sums)

#### Remove element

> Given an array, and a number t, remove all occurrences of t.

Use two pointers i and j: i to incrementally adding elements not equal t, and j for traverse the array.

#### Plus one

> Given a number represented as an array of digits, plus one to the number.

#### Climbing stairs

> How many ways to climb to the n-th stair, if 1 or 2 for each step.

Fibinnaci array.

#### Single number series

> ...

---

#### Linked list

> Python linked list.
>
> ```python
> # Node class
> class Node:
>   def __init__(self, x):
>     self.val = x
>     self.next = None
> 
> # Linked List: maintain a head pointer which has None as self.val
> llist = Node(None)
> ```
>
> **Double pointer method**
>
> Fast pointer and slow pointer. Slow pointer moves 1 step while fast pointer moves k steps. E.g to find the middle point of a linked list, set k to 2, while the fast pointer reaches the end, the slow pointer reaches the 1/2 of the linked list.

##### Sort list

> Sort a linked list in `O(nlogn)` time using constant space complexity.



---

#### Python Data Structure

> Using `list` as `stack` (last in first out or LIFO), i.e. using the `pop()` method.
>
> ```python
> stack = [3, 4, 5]
> stack.append(6)
> stack.append(7)
> # [3, 4, 5, 6, 7]
> stack.pop()
> # [3, 4, 5, 6]
> stack.pop()
> # [3, 4, 5]
> ```

> Using `list` as `queue` (first in first out or FIFO).
>
> "lists are not efficient for FIFO, while appends and pops from end of list are fast, doing inserts or pops from the beginning of a list is slow (because all of the other elements have to be shifted by one)".
>
> Therefore, recommend using `collections.deque`, which is designed to have fast appends and pops from both ends.
>
> ```python
> from collections import deque
> a_list = ['Eric', 'John', 'Michael']
> queue = deque(a_list)
> queue.append('Terry')
> queue.append('Graham')
> queue.popleft()
> # 'Eric'
> queue.popleft()
> # 'John'
> queue
> # ['Michael', 'Terry', 'Graham']
> ```

### Day 5

Today's topic is binary tree. The `python` data structure of binary tree is:

```python
class TreeNode:
  def __init__(self, x):
    self.val = x
    self.left = None
    self.right = None
```

**How to traverse a binary tree?**

- Deep-First traverse
  - Root-first: equals to binary-tree pre-order traverse (先序遍历)
  - Root-last: equals to binary-tree in-order traverse (中序遍历)

- Breath-First traverse

- 前序遍历：根节点-左子树-右子树

- 中序遍历：左子树-根节点-右子树

- 后序遍历：左子树-右子树-根节点

  > 先/中/后：意味着根被先、中、后访问，相对于左-右子树而言；

---

> Binary tree preorder traversal.

```python
# using stack time O(n), space O(n)
result = []
stack = []

if root != None:
  stack.append(root)

while len(stack) != 0:
  node = stack.pop()
  result.append(node.val)
  # right first in stack, then left in stack
  if node.right != None:
    stack.append(node.right)
  if node.left != None:
    stack.append(node.left)

return result
```

```python
# Morris time O(n), space O(1)
result = []
cur = root
prev = None

while cur != None:
  if cur.left != None:
    
```

> Binary tree inorder traversal.

```python
# recursive solution
def visit(root, result):
  if root.left:
    visit(root.left, result)
  result.append(root.val)
  if root.right:
    visit(root.right, result)
```

```python
# stack
result = []
stack = []
p = root
while len(stack) != 0 or p != None:
  if p:
    stack.append(p)
    p = p.left
  else:
    p = stack.pop()
    
```

### Day 6

Today's topic is sorting algorithms.

Basic sorting algorithms:

1. Selection sort

   ```python
   # Insertion sort
   
   ```

2. Bubble sort

   ```python
   # Bubble sort
   def bubble_sort(nums):
     for i in range(len(nums) - 2, 0):
       for j in range(0, i):
         if nums[j] > nums[j + 1]:
           c = nums[j + 1]
           nums[j + 1] = nums[j]
           nums[j] = c
   ```

3. Insertion sort

Advanced sorting algorithms:

1. Merge sort
2. Quick sort
3. Heap sort
4. Bucket sort

