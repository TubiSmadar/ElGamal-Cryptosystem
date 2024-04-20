import random

class PrivateKey(object):
	def __init__(self, p=None, g=None, x=None, numBits=0):
		self.p = p
		self.g = g
		self.x = x
		self.numBits = numBits

class PublicKey(object):
	def __init__(self, p=None, g=None, h=None, numBits=0):
		self.p = p
		self.g = g
		self.h = h
		self.numBits = numBits

def gcd(a, b):
    # """
    #     Computes GCD(greatest common divisor)

    #     Args: 
    #         a: Integer
    #         b: Integer
            
    #     Returns:
    #         GCD(a,b)
    # """
    if b > a:
        t = b
        b = a
        a = t
    while b != 0:
        c = a % b
        a = b
        b = c 
	#a is returned if b == 0
    return a

def primeTest( n, k ):
    # """
    # Performs a Miller-Rabin primality test on a number n with k rounds.

    # Args:
    #     n: The number to test for primality.
    #     k: The number of rounds to perform (higher k indicates more accuracy).

    # Returns:
    #     True if n is likely prime, False if n is composite.
    # """

    # Handle base cases (n <= 1 or n is even)
    if n <= 1 or n % 2 == 0:
        return False
    if n <= 3:
        return True

    # Find d such that n - 1 = 2^s * d
    d = (n - 1) // 2
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(k):
        # Pick a random integer a in the range (2, n-2)
        a = random.randint(2, n - 2)

        # Calculate x = a^d mod n
        x = pow(a, d, n)

        # Check if x == 1 or x == n-1
        if x == 1 or x == n - 1:
            continue

        # Perform loop (checking x becomes n-1 after some iterations)
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break

        # If x is not n-1 after the loop, n is composite
        if x != n - 1:
            return False

    # If all k rounds pass, n is probably prime
    return True

def find_primitive_root(p):
    # """
    # Find a primitive root modulo p.

    # Args:
    #     p (int): The prime number for which to find the primitive root.

    # Returns:
    #     int: A primitive root modulo p.
    # """

    # Base case: p = 2
    if p == 2:
        return 1

    # Calculate p-1 and (p-1)/2
    p_minus_1 = p - 1
    p_half = p_minus_1 // 2

    # Test random g's until one is found that is a primitive root mod p
    while True:
        g = random.randint(2, p_minus_1)

        # Check if g is a primitive root mod p
        if pow(g, p_half, p) != 1 and pow(g, p_minus_1 // p_half, p) != 1:
            return g
		

def find_prime(numBits, confidence=64):
    # """
    # Generate a prime number with the specified bit length.

    # Args:
    #     numBits (int): The bit length of the prime number.
    #     confidence (int): The number of iterations for the Miller-Rabin primality test. Default is 64.

    # Returns:
    #     int: A prime number.
    # """

    while True:
        # Generate a potential prime randomly
        p = random.randint(2**(numBits - 2), 2**(numBits - 1))

        # Ensure the potential prime is odd
        while p % 2 == 0:
            p = random.randint(2**(numBits - 2), 2**(numBits - 1))

        # Test primality using Miller-Rabin test
        if primeTest(p, confidence):
            # Check if p*2 + 1 is also prime
            q = p * 2 + 1
            if primeTest(q, confidence):
                return q
            

def encode(plainText, numBits):
    # """
    # Encodes a plain text string into integers modulo p (assumed to be a prime number).

    # Args:
    #     plainText (str): The plain text string to encode.
    #     numBits (int): The total number of bits used for the prime modulus (p).

    # Returns:
    #     list: A list of integers representing the encoded message (mod p).
    # """

    # Convert the plain text string into a UTF-16 byte array
    byte_array = bytearray(plainText, 'utf-16')

    # Initialize an empty list to store the encoded integers modulo p
    z = []

    # Calculate the number of message bytes needed for each encoded integer
    k = numBits // 8

    # Initialize j to mark the jth encoded integer (start at -k)
    j = -1 * k

    # Initialize num to store the summation of the message bytes
    num = 0

    # Iterate through the byte array
    for i in range(len(byte_array)):
        # Start a new encoded integer if i is divisible by k
        if i % k == 0:
            j += k
            z.append(0)

        # Add the byte multiplied by 2 raised to a multiple of 8
        z[j // k] += byte_array[i] * (2**(8 * (i % k)))

    # Return the array of encoded integers
    return z

def decode(plainText, numBits):
    # """
    # Decodes a list of integers (mod p) back to the original plain text string.

    # Args:
    #     plainText (list): A list of integers representing the encoded message (mod p).
    #     numBits (int): The total number of bits used for the prime modulus (p).

    # Returns:
    #     str: The decoded plain text string.
    # """

    # Calculate the number of bytes per encoded integer
    k = numBits // 8

    # Initialize an empty list to hold the decoded message bytes
    decoded_bytes = []

    # Iterate through each encoded integer in plainText
    for num in plainText:
        # Extract k message bytes from the integer
        for i in range(k):
            # Extract the message byte (8 bits) from the integer
            byte = (num >> (8 * i)) & 0xFF
            # Add the message byte to the decoded bytes list
            decoded_bytes.append(byte)

    # Convert the decoded bytes list to a UTF-16 encoded string
    decoded_text = bytearray(decoded_bytes).decode('utf-16')

    return decoded_text

def generate_keys(numBits=256, confidence=32):
    # """
    # Generates public and private keys.

    # Args:
    #     numBits (int): The length of the keys in bits. Default is 256.
    #     confidence (int): The confidence level for prime number generation. Default is 32.

    # Returns:
    #     dict: A dictionary containing the generated public and private keys.
    #           The keys 'publicKey' and 'privateKey' map to PublicKey and PrivateKey objects respectively.
    # """
    
    # Generate a prime number with the specified bit length and confidence level
    p = find_prime(numBits, confidence)
    
    # Find a primitive root modulo p
    g = find_primitive_root(p)
    
    # Make g a quadratic residue modulo p
    g = pow(g, 2, p)
    
    # Generate a random integer x in the range (0, p-1) inclusive
    x = random.randint(1, (p - 1) // 2)
    
    # Calculate h = g^x mod p
    h = pow(g, x, p)
    
    # Create PublicKey object containing p, g, h, and key length
    publicKey = PublicKey(p, g, h, numBits)
    
    # Create PrivateKey object containing p, g, x, and key length
    privateKey = PrivateKey(p, g, x, numBits)
    
    # Return a dictionary containing the generated public and private keys
    return {'privateKey': privateKey, 'publicKey': publicKey}


def encrypt(key, plainText):
    # """
    # Encrypts a plaintext string using the provided key.

    # Args:
    #     key (PrivateKey): The private key used for encryption.
    #     plainText (str): The plaintext string to be encrypted.

    # Returns:
    #     str: The encrypted ciphertext string.
    # """
    
    # Encode the plaintext string into a list of integers modulo p
    z = encode(plainText, key.numBits)

    # Initialize a list to hold pairs (c, d) corresponding to each integer in z
    cipher_pairs = []

    # Iterate over each integer in z
    for i in z:
        # Pick a random integer y from (0, p-1) inclusive
        y = random.randint(0, key.p)

        # Compute c = g^y mod p
        c = pow(key.g, y, key.p)

        # Compute d = i * h^y mod p
        d = (i * pow(key.h, y, key.p)) % key.p

        # Add the pair (c, d) to the cipher_pairs list
        cipher_pairs.append([c, d])

    # Convert the cipher_pairs list to a string format
    encryptedStr = " ".join([f"{pair[0]} {pair[1]}" for pair in cipher_pairs])

    return encryptedStr

def decrypt(key, cipher):
    # """
    # Decrypts a ciphertext string using the provided key.

    # Args:
    #     key (PrivateKey): The private key used for decryption.
    #     cipher (str): The ciphertext string to be decrypted.

    # Returns:
    #     str: The decrypted plaintext string.
    # """
    
    # Initialize an empty list to store plaintext integers
    plaintext = []

    # Split the ciphertext string into an array of integers
    cipherArray = cipher.split()

    # Check if the length of the cipherArray is even
    if not len(cipherArray) % 2 == 0:
        return "Malformed Cipher Text"

    # Iterate over the cipherArray in pairs
    for i in range(0, len(cipherArray), 2):
        # Extract the pair of integers (c, d)
        c = int(cipherArray[i])  # First number in the pair
        d = int(cipherArray[i + 1])  # Second number in the pair

        # Compute s = c^x mod p
        s = pow(c, key.x, key.p)

        # Compute the plaintext integer = d * s^(-1) mod p
        plain = (d * pow(s, key.p - 2, key.p)) % key.p

        # Add the plaintext integer to the list of plaintext integers
        plaintext.append(plain)

    # Decode the list of plaintext integers to obtain the decrypted plaintext string
    decryptedText = decode(plaintext, key.numBits)

    # Remove trailing null bytes from the decrypted text
    decryptedText = "".join([ch for ch in decryptedText if ch != '\x00'])

    return decryptedText

def test():
		keys = generate_keys()
		private = keys['privateKey']
		public = keys['publicKey']
		message = "This is the message that i want to encrypt, if the decryption will be good i should get the same message in plain variable."
		cipher = encrypt(public, message)
		plain = decrypt(private, cipher)
		print(plain == message)

		return message == plain
if __name__ == '__main__':
	test()