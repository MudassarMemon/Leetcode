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