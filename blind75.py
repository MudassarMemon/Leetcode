#217

class Solution(object):
    def containsDuplicate(self, nums):
        dict = {}

        for num in nums:
            if num in dict:
                return True
            dict[num] = True
        
        return False


#242:
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

#1

class Solution(object):
    def twoSum(self, nums, target):
        dict = {}

        for i in range(len(nums)):
            if (target - nums[i]) in dict:
                return [dict[target-nums[i]], i]
            else:
                dict[nums[i]] = i

     


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
        
# 15
# iteration (ineffecient approach)

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

# effecient 2 pointer solution

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


# 3. ineffecient solution 

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

# effecient solution

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

# 424. ineffecient

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

# 424. effecient solution

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
