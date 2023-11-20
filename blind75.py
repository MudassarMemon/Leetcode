# The start of my leetcode journey. This script/repo is meant to log my DSA progress. It will likely be unorganized at times and may contain multiple tried solutions for the same problems.
# I will revisit and organize as this file grows.
# -- Mudassar Memon


#217

class Solution(object):
    def containsDuplicate(self, nums):
        dict = {}

        for num in nums:
            if num in dict:
                return True
            dict[num] = True
        
        return False


#242

class Solution(object):
    def isAnagram(self, s, t):
        counter = {}

        for i in s: 
            if i in counter:
                counter[i] += 1
            else:
                counter[i] = 1

        for j in t:
            if j not in counter:
                return False
            else:
                counter[j] -= 1
            
        for count in counter.values():
            if count != 0:
                return False
        
        return True

#1 (suboptimal)

class Solution(object):
    def twoSum(self, nums, target):
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i == j:
                    continue
                elif nums[i] + nums[j] == target:
                    return [i, j]

#1 (optimal)

class Solution(object):
    def twoSum(self, nums, target):
        dict = {}

        for i in range(len(nums)):
            if (target - nums[i]) in dict:
                return [dict[target-nums[i]], i]
            else:
                dict[nums[i]] = i

# or

class Solution(object):
    def twoSum(self, nums, target):
        res = {}

        for i, n in enumerate(nums):
            if (target-n) in res:
                return [i, res[target-n]]
            else:
                res[n] = i

     
# 49

class Solution(object):
    def groupAnagrams(self, strs):
        result = {}

        for str in strs:
            str_sort = "".join(sorted(str))
            if str_sort in result:
                result[str_sort].append(str)
            else:
                result[str_sort] = [str]

        return result.values()

#49 (neetcode)

class Solution(object):
    def groupAnagrams(self, strs):
        res = defaultdict(list)

        for str in strs:
            count = [0] * 26

            for char in str:
                count[ord(char) - ord("a")] += 1
            
            res[tuple(count)].append(str)
        
        return res.values()

# 49 (optimal)

class Solution(object):
    def groupAnagrams(self, strs):
        res = defaultdict(list)

        for str in strs:
            str_sorted = "".join(sorted(str))
            res[str_sorted].append(str)
        
        return res.values()

#238

class Solution(object):
    def productExceptSelf(self, nums):
        result = [1] * len(nums)
        prefix = 1
        postfix = 1

        for i in range(len(nums)):
            result[i] = prefix
            prefix *= nums[i]

        for j in range(len(nums)-1, -1, -1):
            result[j] *= postfix 
            postfix *= nums[j]

        return result

#238 (my solution)

class Solution(object):
    def productExceptSelf(self, nums):
        res =  [1] * len(nums)
        prefix = 1
        postfix = 1

        for i in range(len(nums)-1):
            prefix *= nums[i]
            res[i+1] *= prefix

        for j in range(len(nums)-1, 0, -1):
            postfix *= nums[j]
            res[j-1] *= postfix

        return res


#347 


class Solution(object):
    def topKFrequent(self, nums, k):
        result = {}

        for num in nums:
            if num in result:
                result[num] += 1
            else:
                result[num] = 1

        sorted_result = sorted(result, key=result.get, reverse=True)

        return sorted_result[0:k]

# or

class Solution(object):
    def topKFrequent(self, nums, k):
        res = {}

        for num in nums:
            res[num] = 1 + res.get(num, 0)
        
        return sorted(res, key=res.get, reverse=True)[0:k]

# 125

class Solution(object):
    def isPalindrome(self, s):
        s = s.lower()
        l, r = 0, len(s) - 1

        while l < r:

            while l < r and not self.alphaNum(s[l]):
                l += 1
            
            while l < r and not self.alphaNum(s[r]):
                r -= 1

            if s[l] != s[r]:
                return False

            l += 1
            r -= 1
        
        return True


    def alphaNum(self, i):
        return ('a' <= i <= 'z' or '0' <= i <= '9')
        
        """
        :type s: str
        :rtype: bool
        """
        
# 15 (iteration suboptimal)

class Solution(object):
    def threeSum(self, nums):
        nums = sorted(nums)
        result = []

        for i in range(len(nums)-2):
            for j in range(i+1,len(nums) - 1):
                for k in range(j+1,len(nums)):
                    if (nums[i] + nums[j] + nums[k] == 0):
                        arr = [nums[i], nums[j], nums[k]]
                        if arr not in result:
                            result.append(arr)


        return result

# (optimal 2 pointer solution)

class Solution(object):
    def threeSum(self, nums):
        res = []
        nums.sort()

        for i, a in enumerate(nums):
            if i > 0 and a == nums[i-1]:
                continue

            l, r = i + 1, len(nums) - 1

            while l < r:
                three_sum = a + nums[l] + nums[r]

                if (three_sum > 0):
                    r -= 1
                elif (three_sum < 0):
                    l += 1
                else:
                    res.append([a, nums[l], nums[r]])
                    l += 1
                    while (nums[l] == nums[l-1] and l < r):
                        l += 1

        return res

# 11

class Solution(object):
    def maxArea(self, height):
        l, r = 0, len(height) - 1
        res = 0

        while l < r:
            area = (r - l) * min(height[l], height[r])
            if area > res:
                res = area

            if height[l] < height[r]:
                l += 1
            else:
                r -= 1

        return res

# 121

class Solution(object):
    def maxProfit(self, prices):
        max, l, r = 0, 0 , 1

        while r < len(prices):
            profit = prices[r] - prices[l]
            if profit > 0:
                if max < profit:
                    max = profit
            else:
                l = r
                
            r += 1
            
        return max


# 3. (suboptimal) 

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        cur = []
        res = []

        for i in range(len(s)):
            j = i
            while (j < len(s) and s[j] not in cur):
                cur.append(s[j])
                j += 1
            
            if len(cur) > len(res):
                res = cur
            
            cur = []

        return len(res)

#3 (optimal)

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        charSet = set()
        l = 0
        res = 0

        for r in range(len(s)):
            while s[r] in charSet:
                charSet.remove(s[l])
                l += 1

            charSet.add(s[r])
            res = max(res, r - l + 1)

        return res

# 424. (inefficient)

class Solution(object):
    def characterReplacement(self, s, k):
        counter = 0
        res = 0
        arr = []
        l,r = 0, 0

        while r < len(s):
            if len(arr) == 0 or s[r] in arr:
                arr.append(s[r])
                r += 1
            elif counter < k:
                arr.append(True)
                counter += 1
                r += 1
            else:
                counter = 0
                arr = []
                l += 1
                r = l

            res = min(max(res, len(arr)+(k-counter)), len(s))

        return res

# 424. (optimal)

class Solution(object):
    def characterReplacement(self, s, k):
        count = {}
        res = 0
        l = 0

        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)

            while (r - l + 1) - max(count.values()) > k:
                count[s[l]] -= 1
                l += 1
            
            res = max(res, r - l + 1)

        return res



# 424. optimal solution

class Solution(object):
    def characterReplacement(self, s, k):
        count = {}
        maxF = 0
        res = 0
        l = 0

        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            maxF = max(maxF, count[s[r]])

            while (r - l + 1) - maxF > k:
                count[s[l]] -= 1
                l += 1
            
            res = max(res, r - l + 1)

        return res

# 76. optimal

class Solution(object):
    def minWindow(self, s, t):
        if t == "": return ""

        window, hashT = {}, {}

        for i in t:
            hashT[i] = 1 + hashT.get(i, 0)

        res, resLen = [-1, -1], float("infinity")
        l = 0
        have, need = 0, len(hashT)

        for r in range(len(s)):
            c = s[r]
            window[c] = 1 + window.get(c, 0)

            if c in hashT and window[c] == hashT[c]:
                have += 1

            while have == need:
                if (r - l + 1 < resLen):
                    resLen = r - l + 1
                    res = [l, r]
                    
                window[s[l]] -= 1

                if s[l] in hashT and window[s[l]] < hashT[s[l]]:
                    have -= 1
                    
                l += 1

        
        l, r = res

        return s[l:r+1] if resLen != float("infinity") else  ""
        

# 20. suboptimal

class Solution(object):
    def isValid(self, s):
        hash = {"(": ")", "{": "}", "[": "]"}
        temp = []

        for i in s:
            if i in hash.keys():
                temp.append(hash[i])
            elif temp and temp[-1] == i:
                temp.pop()
            else:
                return False

        if len(temp) == 0: 
            return True
        else:
            return False


# 153 suboptimal:

class Solution(object):
    def findMin(self, nums):
        res = float("infinity")

        for n in nums:
            if n < res:
                res = n
        
        return res

# 12 (my solution - time efficient)

class Solution(object):
    def intToRoman(self, num):
        convert = {1: "I", 5: "V", 10: "X", 50: "L", 100: "C", 500: "D", 1000: "M", 4: "IV", 9: "IX", 40: "XL", 90: "XC", 400: "CD", 900: "CM"}
        keys = sorted(convert, reverse=True)
        res = ""

        for key in keys:
            while(num >= key):
                res += convert[key]
                num -= key
                if num == 0:
                    return res

# 12 (my solution - space efficient)

class Solution(object):
    def intToRoman(self, num):
        convert = {1: "I", 5: "V", 10: "X", 50: "L", 100: "C", 500: "D", 1000: "M", 4: "IV", 9: "IX", 40: "XL", 90: "XC", 400: "CD", 900: "CM"}
        res = ""

        for key in sorted(convert, reverse=True):
            while(num >= key):
                res += convert[key]
                num -= key
                if num == 0:
                    return res

# 153 (my solution - recursive)

class Solution(object):
    def findMin(self, nums):
        n = len(nums)

        if n == 1:
            return nums[0]

        mid = n//2

        if nums[0] < nums[-1]:
            if nums[0] < nums[mid]:
                return self.findMin(nums[:mid])
            else:
                return self.findMin(nums[mid:])
        elif nums[-1] < nums[mid-1]:
            return self.findMin(nums[mid:])
        else:
            return self.findMin(nums[:mid])
        

# 153 (not recursive - optimal)

class Solution(object):
    def findMin(self, nums):
        l, r = 0, len(nums) - 1
        res = nums[0]

        while l <= r:
            if nums[l] <= nums[r]:
                return min(res,nums[l])
            
            m = (l+r+1)//2

            if nums[l] < nums[m]:
                l = m+1
            else:
                l += 1
                r = m

# 167 (my solution)

class Solution(object):
    def twoSum(self, numbers, target):
        l, r = 0, len(numbers) - 1

        while l < r:
            twosum = numbers[l] + numbers[r]
            if twosum == target:
                return [l+1, r+1]
            elif twosum > target:
                r -= 1
            else:
                l += 1

# 31. (my solution)

class Solution(object):
    def search(self, nums, target):
        l, r = 0, len(nums) - 1

        while l <= r:
            m = (l + r + 1) // 2
            if (r - l + 1) <= 2:
                if nums[l] == target:
                    return l
                elif nums[r] == target:
                    return r
                else:
                    return -1

            elif nums[l] < nums[m]:
                if nums[l] <= target <= nums[m]:
                    r = m
                else:
                    l = m + 1
            elif nums[l] > nums[m]:
                if target >= nums[l] or target <= nums[m]:
                    r = m
                else:
                    l = m + 1
            

        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        

# 20. (my solution)

class Solution(object):
    def isValid(self, s):
        stack = []

        for c in s:
            if c == "{":
                stack.append("}")
            elif c == "[":
                stack.append("]")
            elif c == "(":
                stack.append(")")
            elif len(stack) == 0:
                return False
            elif len(stack) > 0 and c != stack.pop():
                return False
            
            
        return True if len(stack) == 0 else False

        """
        :type s: str
        :rtype: bool
        """
        
# 735 (my solution)
class Solution(object):
    def asteroidCollision(self, asteroids):

        done = False

        while not done:
            done = True
            for i in range(len(asteroids) - 1):
                if asteroids[i] < 0:
                    continue
                else:
                    if asteroids[i+1] > 0:
                        continue
                    elif asteroids[i] > asteroids[i+1] / -1:
                        asteroids.pop(i+1)
                    elif asteroids[i] < asteroids[i+1] / -1:
                        asteroids.pop(i)
                    else:
                        asteroids.pop(i)
                        asteroids.pop(i)
                    done = False
                    break
                        

        return asteroids

# 56. (my solution)

class Solution(object):
    def merge(self, intervals):
        intervals.sort()
        res = []

        for i in intervals:
            if res:
                if i[0] <= res[-1][1]:
                    res[-1][1] = max(res[-1][1], i[1])
                else:
                    res.append(i)
            else:
                res.append(i)
            
        return res
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """

# 209. (my solution)

class Solution(object):
    def minSubArrayLen(self, target, nums):
        l,r = 0, 0 
        res = float("infinity")
        total = nums[l]
        count = 1
        
        while l <= r and r < len(nums):
            

            while total < target and r < len(nums) - 1:
                r += 1
                count += 1
                total += nums[r]

                if count >= res:
                    break

            if total >= target:
                res = min(res, count)
                if res == 1:
                    return res
            elif r == len(nums) - 1:
                break
                

            total -= nums[l]
            count -= 1
            l += 1        

        if res != float("infinity"):
            return res
        else:
            return 0

        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        

# 219. (my solution - suboptimal

class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        l = 0
        count = Counter(nums)
        print(count)

        while l < len(nums) - 1:
            if count[nums[l]] == 1:
                l += 1
                continue

            r = l + 1

            while r < len(nums) and r - l <= k:
                if nums[l] == nums[r]:
                    return True
                else:
                    r += 1
            l += 1

        return False

        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        
# 219. optimal 

class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        l = 0
        window = set()

        for r in range(len(nums)):
            if r - l > k:
                window.discard(nums[l])
                l += 1
            
            if nums[r] in window:
                return True
                
            else:
                window.add(nums[r])

        return False

        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        
# 206. recursive

class Solution(object):
    def reverseList(self, head):

        if not head:
            return None

        if head.next:
            newHead = self.reverseList(head.next)
            head.next.next = head
            head.next = None
        else:
            return head

        return newHead

# 206. Iterative:

class Solution(object):
    def reverseList(self, head):
        prev, curr = None, head

        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt

        return prev

# 21. 

class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """

        res = ListNode()
        tail = res
        

        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next

        if list1:
            tail.next = list1
        else:
            tail.next = list2

        return res.next

# 143.

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """

        slow, fast = head, head.next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next


        second = slow.next
        slow.next = None
        prev = None

        while second:
            nxt = second.next
            second.next = prev
            prev = second
            second = nxt

        first = head
        second = prev

        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first = tmp1
            second = tmp2


        
        
# 19. (my solution)

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeNthFromEnd(self, head, n):
        
        m = 1
        tail = head.next

        while m != n:
            tail = tail.next
            m += 1

        curr = head
        prev = None
        if not tail:
            head = curr.next
            return head

        while tail:
            prev = curr
            curr = curr.next
            tail = tail.next


        prev.next = curr.next

        return head


        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        

# 141. (my solution)

        # Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):

        res = set()
        i = 0

        while head not in res:
            res.add(head)
            if not head or not head.next:
                return False
            head = head.next

        return True


        """
        :type head: ListNode
        :rtype: bool
        """

        
# 141. Tortoise and Hare solution

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow, fast = head, head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True

        
        return False


# 23. (using merge sort)

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeKLists(self, lists):

        if not lists or len(lists) == 0:
            return None

        while len(lists) > 1:
            merged = []

            for i in range(0, len(lists), 2):
                list1 = lists[i]
                list2 = lists[i+1] if (i+1 < len(lists)) else None
                merged.append(self.mergeLists(list1, list2))
            lists = merged

        return lists[0]

    
    def mergeLists(self, list1, list2):
        mergedNodeList = ListNode()
        tail = mergedNodeList

        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        

        if list1:
            tail.next = list1
        else:
            tail.next = list2

        return mergedNodeList.next




        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """

# 226.

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def invertTree(self, root):

        if not root:
            return None

        temp = root.left
        root.left = root.right
        root.right = temp

        self.invertTree(root.left)
        self.invertTree(root.right)

        return root

        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        

# 104. Recursive DFS:

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if not root:
            return 0

        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

# 104. Iterative BFS:

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if not root:
            return 0

        queue = deque([root])
        depth = 0

        while queue:
            for i in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            depth += 1

        return depth


# 104. Iterative DFS

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        

        stack = [[root, 1]]
        res = 0

        while stack:
            node, depth = stack.pop()

            if node:
                res = max(res, depth)
                stack.append([node.left, depth + 1])
                stack.append([node.right, depth + 1])

        return res
    
# 100. Recursive DFS

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSameTree(self, p, q):
        if not p and not q:
            return True
        
        if not p or not q or p.val != q.val:
            return False

        return (self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right))

        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """

572.

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSubtree(self, root, subRoot):
        if not subRoot:
            return True
        if not root:
            return False
        
        if self.isSame(root, subRoot):
            return True

        return (self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot))

        
    def isSame(self, t, s):
        if not t and not s:
            return True
        if not t or not s or s.val != t.val:
            return False
        
        return (self.isSame(t.left, s.left) and self.isSame(t.right, s.right))

        """
        :type root: TreeNode
        :type subRoot: TreeNode
        :rtype: bool
        """

# 235. 

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        curr = root

        while curr:
            if p.val > curr.val and q.val > curr.val:
                curr = curr.right
            elif p.val < curr.val and q.val < curr.val:
                curr = curr.left
            else:
                return curr
        

        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        