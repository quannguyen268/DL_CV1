#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Convert hexa to binary
def hex_to_bin(hex):
    n = int(hex, 16)
    bin = [] 
    while n > 0: 
        bin.append((n % 2))
        n  >>= 1
    while len(bin) < (4*len(hex)):
        bin.append(0)
    return bin[::-1]

#Convert binary to hexa
def bin_to_hex(bin):
    return hex(int(bin, 2))[2:]

#Convert decimal to binary
def dec_to_bin(dec):
    bin = [] 
    while dec > 0: 
        bin.append((dec % 2))
        dec  >>= 1
    while len(bin) < 4:
        bin.append(0)
    return bin[::-1]

def bin_to_string(bin):
    string_bit = ''.join(str(i) for i in bin )
    return bin_to_hex(string_bit)

#XOR function
def xor(x1, x2):
    return [x^y for x,y in zip(x1,x2)]

#Left shift
def shift(c, d, i): 
    return c[i:] + c[:i], d[i:] + d[:i]

#Permut block using given table 
def permute(block, matrix):
    return [block[i-1] for i in matrix]

# split code into 2 part 
def split_code(s, n):
    return [s[k:k+n] for k in range(0, len(s), n)]

def add_bit(plaintext, key):

    bit_len1 = 16 - (len(plaintext) % 16)
    plaintext = bit_len1 * "0" + plaintext
    print(plaintext)
add_bit("123456789acbd", "123456789aaaaaa")


    # bit_len2 = 16 - (len(key) % 16)
    # key += bit_len2 * "0"

encrypt = 1
decrypt = 0


# In[6]:


#Initial permut matrix 
IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

# Permute for initial key
PC_1 = [57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4]

#Permut after shifted key to get Ki+1
PC_2 = [14, 17, 11, 24, 1, 5, 3, 28,
        15, 6, 21, 10, 23, 19, 12, 4,
        26, 8, 16, 7, 27, 20, 13, 2,
        41, 52, 31, 37, 47, 55, 30, 40,
        51, 45, 33, 48, 44, 49, 39, 56,
        34, 53, 46, 42, 50, 36, 29, 32]

# matrix E to get 48bits block from 32bit Ri to apply the xor with Ki
E = [32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9, 10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1]

#SBOX
S_BOX = [
         
[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
 [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
 [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
 [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
],

[[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
 [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
 [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
 [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
],

[[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
 [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
 [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
 [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
],

[[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
 [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
 [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
 [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
],  

[[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
 [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
 [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
 [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
], 

[[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
 [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
 [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
 [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
], 

[[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
 [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
 [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
 [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
],
   
[[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
 [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
 [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
 [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
]
]

#Permut made after each SBox substitution for each round
P = [16, 7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26, 5, 18, 31, 10,
     2, 8, 24, 14, 32, 27, 3, 9,
     19, 13, 30, 6, 22, 11, 4, 25]

#Final permut for datas after the 16 rounds
IP_1 = [40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25]

#Matrix that determine the shift for each round of keys
SHIFT = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]
        
    


# In[7]:


class DES:
    def __init__(self):
        self.key = None
        self.Ki = []
        self.plaintext = None
     
    # Generate 16 sub keys 48bit
    def generate_key(self):
        key = hex_to_bin(self.key)
        print("Initial key [{}]: ".format(self.key), key, "\n")
        # Use PC-1 matrix to permute initial key
        key = permute(key, PC_1)
        print("Key after permute with PC-1 [{}]:".format(bin_to_string(key)), key, "\n")
        c,d = split_code(key, 28)
        for i in range(16):
            # Left shift 16 times
            c,d = shift(c,d, SHIFT[i])
            #merge Ci and Di
            Ci_Di = c + d
            self.Ki.append(permute(Ci_Di, PC_2))
            print("key {} [{}]:{} \n".format((i+1),bin_to_string(self.Ki[i]), self.Ki[i]))
            
    def permute_with_Sbox(self, bit48_matrix):
        #split into 8 block 6bit
        bit6_blocks = split_code(bit48_matrix, 6)
        
        result = []
        
        for i in range(len(bit6_blocks)):
            block = bit6_blocks[i]
            print("Block s{} : {} \n".format(i+1, block))
            row = int(str(block[0])+str(block[5]),2)#Get the row with the first and last bit
            column = int(''.join([str(x) for x in block[1:-1]]),2) #Column is the 2,3,4,5th bits
            temp = S_BOX[i][row][column]
            bin = dec_to_bin(temp)
            result += [int(x) for x in bin]#And append it to the resulting list
        print("After Sbox permute: ", result, "\n")
        return result
    
    # Add bit to plaintext in case not enough 64bit

        
        
    def activate(self, text,key, action=encrypt):
        self.key = key
        self.plaintext = text
        add_bit(self.plaintext, self.key)
        if len(key) > 16 or len(text) > 16:
            key = key[:16]  # If key size is above 64bit, cut to be 64bit
            text = text[:16]
        

        self.plaintext = hex_to_bin(text)
        print(len(self.plaintext))
        
        if action==encrypt:
            print("Initial plaintext [{}]: {}".format(text, self.plaintext), "\n")
        else:
            print("Initial ciphertext [{}]: {}".format(text,self.plaintext), "\n")
        
        #First : generate 16 keys
        self.generate_key() 
        
        #Second : apply IP permute and split plaintext into 2 part 
        plaintext_IP = permute(self.plaintext, IP)
        L, R = split_code(plaintext_IP, 32) # L(left), R(right)
        result = []
        
        print("Plaintext after permute with IP [{}]: {} \n".format(bin_to_string(R+L), R+L))
        print("L0 [{}]: {} \n".format(bin_to_string(L), L))
        print("R0 [{}]: {} \n".format(bin_to_string(R), R))
        temp = None
        
        # Apply Feistel Cipher
        for i in range(16):
            # Apply E to R0 for 48bit block
            R_E = permute(R, E)
            if action == encrypt:
                temp = xor(self.Ki[i], R_E) # R0 after apply E matrix then xor with key[i]
            else :
                temp = xor(self.Ki[15-i], R_E) # R0 after apply E matrix then xor with key[i]
            temp = self.permute_with_Sbox(temp) # Apply Sbox permute for 32bits block
            temp = permute(temp, P) 
            temp = xor(L, temp)
            L = R
            R = temp
            print("L{} [{}]: {} \n".format(i+1,bin_to_string(L), L))
            print("R{} [{}]: {} \n".format(i+1,bin_to_string(R), R))
        
        #Apply last permute IP-1
        if action==encrypt:
            print("Cipher text before permute with IP-1 matrix [{}]: {} \n".format(bin_to_string(R+L),R+L) )
            result = permute(R+L, IP_1)
            print("Cipher text [{}]: {} \n".format(bin_to_string(result), result) )
        else : 
            print("Plain text before permute with IP-1 matrix: {} \n".format(R+L) )
            result = permute(R+L, IP_1)
            print("Plain text [{}]: {} \n".format(bin_to_string(result), result) )
            return result
        
        

            

        


# In[8]:


des = DES()
des.activate("121218ecbbca845","dcba1009abcd0910", action=encrypt)


# In[ ]:


a = "abcd"
print(len(a))


# In[ ]:





# In[ ]:




