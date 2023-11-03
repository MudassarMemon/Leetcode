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
        result = []

        for i in range(len(nums)-2):
            for j in range(i+1,len(nums) - 1):
                for k in range(j+1,len(nums)):
                    if (nums[i] + nums[j] + nums[k] == 0):
                        arr = sorted([nums[i], nums[j], nums[k]])
                        if arr not in result:
                            result.append(arr)


        return result