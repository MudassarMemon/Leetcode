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
        