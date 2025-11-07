class Solution:
    def isPalindrome(self, x: int) -> bool:
        my_string = str(x)
        return True if my_string == my_string[::-1] else False
