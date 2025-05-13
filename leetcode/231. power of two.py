class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if  n==1:
            return True
        elif n==0:
            return False
        binary = bin(n)
        return int(bin(n)[3:]) ==0
