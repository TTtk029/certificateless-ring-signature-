import math
from re import S, T, U
import numpy as np
import time
import hashlib

from numpy.random import f

# Gaussian sampling algorithm
def sample_discrete_gaussian_polynomial(x, q, sigma):
    n = len(x)
    e = np.random.normal(loc=0.0, scale=sigma, size=n)
    e_rounded = np.round(e).astype(int)
    sampled_coeffs = (np.array(x) + e_rounded) % q
    return sampled_coeffs.tolist()

# Generate the unit matrix of dxd
def generate_identity_matrix(d, n):
    identity = np.zeros((d, d, n), dtype=int)
    np.fill_diagonal(identity[..., 0], 1)
    return identity

# Generation of Kronecker products
def kronecker_product(matrix1, matrix2, n, q):
    a, b, _ = matrix1.shape
    c, d, _ = matrix2.shape
    result = np.zeros((a * c, b * d, n), dtype=int)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                for l in range(d):
                    product = poly_multiply(matrix1[i, j], matrix2[k, l], n ,q)
                    result[i * c + k, j * d + l] = product
    return result

# Polynomial multiplication and modulo (X^n +1)
def poly_multiply(poly1, poly2, n, q):
    poly1 = np.array(poly1, dtype=np.int64)
    poly2 = np.array(poly2, dtype=np.int64)
    product = np.convolve(poly1, poly2, mode='full').astype(np.int64)
    
    result = np.zeros(n, dtype=np.int64)
    result[:n] = product[:n]
    for idx in range(n, len(product)):
        result[idx - n] = np.subtract(
            result[idx - n], 
            product[idx], 
            dtype=np.int64
        )
    return (result % q).astype(int)
# matrix multiplication
def matrix_multiply(matrix1, matrix2, q, n):
    rows1, cols1 = matrix1.shape[0], matrix1.shape[1]
    cols2 = matrix2.shape[1]
    result = np.zeros((rows1, cols2, n), dtype=int)
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                product = poly_multiply(matrix1[i, k], matrix2[k, j], n, q)
                result[i, j] = (result[i, j] + product) % q
    return result

# polynomial inverse
def inverse_polynomial(f, q, n):
    constant_term = f[0]
    inv_constant = pow(int(constant_term), -1, q)
    inverse = np.zeros(n, dtype=int)
    inverse[0] = inv_constant
    return inverse.tolist()

# term vector inner product
def poly_inner_product(vec1, vec2, q, n):
    products = [poly_multiply(a, b, n ,q) for a, b in zip(vec1, vec2)]
    sum_product = np.sum(products, axis=0) % q
    return sum_product.tolist()

# Hash_{1}
def hash_to_poly_vector(bitstring: bytes, q: int, n: int, d: int) -> list:

    bytes_per_coeff = (q.bit_length() + 7) // 8
    byte_length = n * d * bytes_per_coeff
    # Scalable output using SHAKE256
    hasher = hashlib.shake_256()
    hasher.update(bitstring)
    hash_bytes = hasher.digest(byte_length)
    # Generate polynomial vectors
    vector = []
    for i in range(d):
        coeffs = []
        for j in range(n):
            start = (i * n + j) * bytes_per_coeff
            end = start + bytes_per_coeff
            coeff = int.from_bytes(hash_bytes[start:end], 'big') % q
            coeffs.append(coeff)
        vector.append([coeffs])
    
    return vector

# Hash_{2}
def hash_to_binary_poly(bitstring: bytes, q: int, n: int) -> list:
    # Calculate the number of bits to be generated (2 bits per factor)
    required_bits = 2 * n
    byte_length = (required_bits + 7) // 8
    
    # Generating random bits using SHAKE256
    hasher = hashlib.shake_256()
    hasher.update(bitstring)
    hash_bytes = hasher.digest(byte_length)
    
    # Converting Bytes to Bitstreams
    bit_stream = ''.join(f"{byte:08b}" for byte in hash_bytes)
    
    # Generate coefficients (mapping rules: 00->0, 01->1, 10->-1, 11->0)
    coefficients = []
    for i in range(0, required_bits, 2):
        bits = bit_stream[i:i+2]
        if bits == '00': coeff = 0
        elif bits == '01': coeff = 1
        elif bits == '10': coeff = -1
        else: coeff = 0  
        coefficients.append(coeff)
    
    return coefficients[:n]  

# Hash_{3}
def hash_to_poly_matrix(bitstring: bytes, q: int, n: int, d: int, m1: int, m2: int) -> list:
    matrix = []
    total_cols = m1 + m2
    bytes_per_coeff = max(1, (q.bit_length() + 7) // 8)
    bytes_per_poly = n * bytes_per_coeff
    bytes_needed = d * total_cols * bytes_per_poly

    # Generating Deterministic Random Bytes with SHAKE256
    hasher = hashlib.shake_256()
    hasher.update(bitstring)
    hash_bytes = hasher.digest(bytes_needed)

    for row in range(d):
        matrix_row = []
        for col in range(total_cols):
            # Calculate the byte start position of the current polynomial
            offset = (row * total_cols + col) * bytes_per_poly
            poly_bytes = hash_bytes[offset:offset+bytes_per_poly]
            
            # Convert bytes to polynomial coefficients
            coeffs = []
            for i in range(n):
                start = i * bytes_per_coeff
                end = start + bytes_per_coeff
                coeff = int.from_bytes(
                    poly_bytes[start:end],
                    'little'
                ) % q
                coeffs.append(coeff)
            matrix_row.append(coeffs)
        matrix.append(matrix_row)
    return matrix

# Sampling binary polynomial vectors
def sample_binary_poly_vector(m2: int, n: int) -> list:
    # Generate a random integer representation of 0-3 with 2 bits (each coefficient is encoded with 2 bits)
    bits_array = np.random.randint(0, 4, size=(m2, n))
    # Mapping bits to coefficients (00→0, 01→1, 10→-1, 11→0)
    coeffs = np.where(bits_array == 1, 1, np.where(bits_array == 2, -1, 0))
    return coeffs.tolist()

# trapdoor generation
def tarpdoor(n, d, k):
    q = 2**k
    g = np.zeros((1, k, n), dtype=np.int64)
    for i in range(k):
        g[0, i, 0] = 2**i
    identity = generate_identity_matrix(d, n)
    G = kronecker_product(identity, g, n, q)
    A_prime = np.random.randint(0, q, size=(d, d, n), dtype=np.int64 )
    R1 = np.random.randint(0, q, size=(d, d*k, n), dtype=np.int64)
    R2 = np.random.randint(0, q, size=(d, d*k, n), dtype=np.int64)
    R = np.vstack((R1, R2))
    A_prime_R2 = matrix_multiply(A_prime, R2, q, n)
    A_prime_R2_minus_R1 = (A_prime_R2 + R1) % q
    G_minus = (G - A_prime_R2_minus_R1) % q
    I_d = generate_identity_matrix(d, n)
    A = np.concatenate((I_d, A_prime, G_minus), axis=1)
    return A, R

# primary image sampling
def Sample_Pre(n, d, k, R, sigma, f_ID_i):
    q = 2**k
    u = f_ID_i
    u = np.array(u)
    u = np.array(u)
    g = np.zeros((1, k, n), dtype=int)
    for i in range(k):
        g[0, i, 0] = 2**i
    all_X = np.zeros((k, d, n), dtype=int)
    for i in range(d):
        u1 = u[i, 0].tolist()
        selected_i = next(j for j, gi in enumerate(g[0]) if gi[0] % 2 != 0)
        g_inv = inverse_polynomial(g[0, selected_i], q, n)
        product = poly_multiply(u1, g_inv, n, q)
        all_X[selected_i, i] = np.array(product) % q
    all_neg_X = (-all_X) % q
    Sk = np.zeros((k, k, n), dtype=int)
    for i in range(k):
        Sk[i, i, 0] = 2
        if i < k-1:
            Sk[i+1, i, 0] = -1
    Sk_prime = np.zeros((k, k, n), dtype=int)
    np.fill_diagonal(Sk_prime[..., 0], 2)
    e = []
    for ii in range(d):
        t = all_neg_X[:, ii]
        v = np.zeros((k, n), dtype=int)
        for i in reversed(range(1, k)):
            Sk_col = Sk_prime[:, i]
            t_prime = poly_inner_product(t, Sk_col, q, n)
            t_prime = [x//4 for x in t_prime]
            z = sample_discrete_gaussian_polynomial(t_prime, q, sigma)
            Sk_col_orig = Sk[:, i]
            z_Sk = [poly_multiply(z, col, n, q) for col in Sk_col_orig]
            t = (t - z_Sk) % q
            v = (v + z_Sk) % q
        e_prime = (all_X[:, ii] + v) % q
        e.extend(e_prime[:, np.newaxis, :])
    e = np.array(e)
    I_dk = generate_identity_matrix(d*k, n)
    y_prime = np.vstack((R, I_dk))
    y = matrix_multiply(y_prime, e, q, n)
    return  y.tolist()

# Setup
def Setup(n, d, k, m_2):
    q = 2**k
    G_prime = np.random.randint(0, q, size=(d, m_2-d, n), dtype=np.int64)
    I_d = generate_identity_matrix(d, n)
    G = np.concatenate((I_d, G_prime), axis=1)
    A, T_A = tarpdoor(n, d, k)
    return A, T_A, G

# Set-Public-key
def Set_Public_Key(n, d, k, m_2, G, l, ID_Hash1):
    q = 2**k
    Upk_ID_np = []
    s_ID_np = []
    f_ID_np = []
    for i in range(l):
        s_ID_i = sample_binary_poly_vector(m_2, n)
        s_ID_i = np.array(s_ID_i)
        s_ID_i = s_ID_i.reshape(m_2, 1, n)
        p_ID_i = matrix_multiply(G, s_ID_i, q, n)
        f_ID_i = (ID_Hash1[i] - p_ID_i) % q
        s_ID_np.append(s_ID_i)
        Upk_ID_np.append(p_ID_i)
        f_ID_np.append(f_ID_i)
    s_ID = np.array(s_ID_np)
    Upk_ID = np.array(Upk_ID_np)
    f_ID = np.array(f_ID_np)
    return s_ID, Upk_ID, f_ID

# Extract-Partial-Private-Key
def Extract_Partial_Private_Key(n, d, k, l, f_ID, A, T_A, sigma):
    q = 2**k
    d_ID_np = []
    for i in range(l):
        f_ID_i = f_ID[i]
        y_i = Sample_Pre(n, d, k, T_A, sigma, f_ID_i)
        d_ID_np.append(y_i)
    d_ID = np.array(d_ID_np)
    return d_ID
        
# Set-Private-Key
def Set_Private_Key(n, d, k, l, d_ID, s_ID):
    q = 2**k
    Usk_ID_np = []
    for i in range(l):
        d_ID_i = d_ID[i]
        s_ID_i = s_ID[i]
        Usk_ID_i = np.vstack((d_ID_i, s_ID_i))
        Usk_ID_np.append(Usk_ID_i)
    Usk_ID = np.array(Usk_ID_np)
    return Usk_ID

# Set-Link-Tag
def Set_Link_Tag(n, d, k, l, A_com, Usk_ID):
    q = 2**k
    v_ID_np = []
    for i in range(l):
        v_ID_i = matrix_multiply(A_com, Usk_ID[i], q, n)
        v_ID_np.append(v_ID_i)
    v_ID = np.array(v_ID_np)
    return v_ID

# Sign
def Sign(n, d, k, m_1, m_2, l, A, G, A_com, v_ID, ID_Hash1, Usk_ID, Upk_ID, sigma, j, mu):
    q = 2**k
    r = np.random.randint(0, (m_1 + m_2) * n**2, size=(m_1 + m_2, 1, n), dtype=int)
    AG = np.hstack((A, G))
    e = matrix_multiply(AG, r, q, n)
    c_np = []
    for i in range(l):
        if i == j:
            c_i = np.zeros((1, n), dtype=int).tolist()
        else:
            c_i = sample_binary_poly_vector(1, n)
        c_np.append(c_i)
    c = np.array(c_np)
    R = np.zeros((d, n), dtype=int)
    for i in range(l):
        R_i = np.empty((0, n), dtype=int)
        for ii in range(d):
            R_ii = poly_inner_product(c[i], ID_Hash1[i][ii], q, n)
            R_i = np.vstack((R_i, R_ii))
        R = (R + R_i) % q
    R = R.reshape(d, 1, n)
    R = (R + e) % q
    A_com_r = matrix_multiply(A_com, r, q, n)
    T = np.zeros((d, n), dtype=int)
    for i in range(l):
        T_i = np.empty((0, n), dtype=int)
        for ii in range(d):
            T_ii = poly_inner_product(c[i], v_ID[j][ii], q, n)
            T_i = np.vstack((T_i, T_ii))
        T = (T + T_i) % q
    T = T.reshape(d, 1, n)
    T = (T + A_com_r) % q
    R_data = R.tobytes()
    T_data = T.tobytes()
    Upk_ID_data = Upk_ID.tobytes()
    mu_data = str(mu).encode()
    H2_date = R_data + T_data + Upk_ID_data + mu_data
    H2 = hash_to_binary_poly(H2_date, q, n)
    H2 = np.array(H2)
    H2 = H2.reshape(1, n)
    c_sum = np.sum(c, axis=0)
    c[j] = (H2 - c_sum) % 2
    z_prime = np.empty((0, n), dtype=int)
    for i in range(m_1 + m_2):
        z_i = poly_inner_product(c[j], Usk_ID[j][i], q, n)
        z_prime = np.vstack((z_prime, z_i))
    z_prime = z_prime.reshape(m_1 + m_2, 1, n)
    z = (z_prime - r) % q
    return z, c

# Verify
def Verify(n, d, k, m_1, m_2, l, A, G, A_com, v_ID, ID_Hash1, Upk_ID, j, z, c, mu):
    q = 2**k
    AG = np.hstack((A, G))
    AG_z = matrix_multiply(AG, z, q, n)
    R = np.zeros((d, n), dtype=int)
    for i in range(l):
        R_i = np.empty((0, n), dtype=int)
        for ii in range(d):
            R_ii = poly_inner_product(c[i], ID_Hash1[i][ii], q, n)
            R_i = np.vstack((R_i, R_ii))
        R = (R + R_i) % q
    R = R.reshape(d, 1, n)
    R = (R - AG_z) % q

    A_com_z = matrix_multiply(A_com, z, q, n)
    T = np.zeros((d, n), dtype=int)
    for i in range(l):
        T_i = np.empty((0, n), dtype=int)
        for ii in range(d):
            T_ii = poly_inner_product(c[i], v_ID[j][ii], q, n)
            T_i = np.vstack((T_i, T_ii))
        T = (T + T_i) % q
    T = T.reshape(d, 1, n)
    T = (T - A_com_z) % q
    R_data = R.tobytes()
    T_data = T.tobytes()
    Upk_ID_data = Upk_ID.tobytes()
    mu_data = str(mu).encode()
    H2_date = R_data + T_data + Upk_ID_data + mu_data
    H2 = hash_to_binary_poly(H2_date, q, n)
    H2 = np.array(H2) % 2
    H2 = H2.reshape(1, n)
    c_sum = np.sum(c, axis=0) % 2
    if np.array_equal(H2, c_sum):
        return True
    else:
        return False


# Setting parameters
k = 35
q = 2**k
n = 128
d = 9
m_1 = (k + 2) * d
m_2 = 45

# Setting the Signer Index
j = 0

# Setting the plaintext
mu = 123545454564
sigma = 2 * math.sqrt(math.log(2 * n * (1 + 1 / (2**(-80))) / math.pi))

# Setting the ID list
ID_list = np.arange(2)
l = ID_list.shape[0]

# Calculate Hash-1 for each member ID
print('Generating Hash-1...')
ID_Hash1_np = []
for i in range(l):
    ID_Hash1_i = hash_to_poly_vector(ID_list[i], q, n, d)
    ID_Hash1_np.append(ID_Hash1_i)
ID_Hash1 = np.array(ID_Hash1_np)

# Generate master public key Msk and master private key Msk
print('Generating master public key Msk and master private key Msk...')
A, T_A, G = Setup(n, d, k, m_2)

# Generate A_com matrix
print('Generating A_com matrix...')
A_com_prime =  np.append(A, G, axis=1)
A_com_prime_date = A_com_prime.tobytes()
A_com = hash_to_poly_matrix(A_com_prime_date, q, n, d, m_1, m_2)
A_com = np.array(A_com)

# Generate user public key Upk_ID and private key s_ID
print('Generating user public key Upk_ID and private key s_ID...')
s_ID, Upk_ID, f_ID = Set_Public_Key(n, d, k, m_2, G, l, ID_Hash1)


# Generate user partial private key d_ID
print('Generating user partial private key d_ID...')
d_ID = Extract_Partial_Private_Key(n, d, k, l, f_ID, A, T_A, sigma)



# Generate user private key Usk_ID
print('Generating user private key Usk_ID...')
Usk_ID = Set_Private_Key(n, d, k, l, d_ID, s_ID)

# Generate user label v_ID
print('Generating user label v_ID...')
v_ID = Set_Link_Tag(n, d, k, l, A_com, Usk_ID)

# Generate a signature about user j
print('Generating a signature about user j...')
start = time.time()
z, c = Sign(n, d, k, m_1, m_2, l, A, G, A_com, v_ID, ID_Hash1, Usk_ID, Upk_ID, sigma, j, mu)
print("Signature generation time:", round((time.time() - start) * 1000, 3), "ms") 


# Verify Signature
print('Verifying Signature...')
start1 = time.time()
result = Verify(n, d, k, m_1, m_2, l, A, G, A_com, v_ID, ID_Hash1, Upk_ID, j, z, c, mu)
print("Signature verification time:", round((time.time() - start1) * 1000, 3), "ms")
if result:
    print("Signature Verification Successful！")
else:
    print("Signature verification failed！")
















