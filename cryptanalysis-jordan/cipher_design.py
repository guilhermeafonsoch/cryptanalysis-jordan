"""
Simple 4-bit Stream Cipher Design for Neural Network Analysis

This module implements a deterministic 4-bit stream cipher with an 8-bit internal state.
The cipher is NOT cryptographically secure, but is designed to be learnable by neural networks.

Architecture:
- Input: 4-bit plaintext nibble (0-15)
- State: 8-bit internal state (0-255)
- Output: 4-bit ciphertext nibble + 8-bit next state
- Mechanism: XOR-based transformation with state update using Fibonacci LFSR

State Table:
- Total transitions: 256 states × 16 input values = 4,096 unique (state, plaintext) → (ciphertext, next_state) pairs
"""

import numpy as np
import pandas as pd


class SimpleCipher:
    """Simple 4-bit stream cipher with 8-bit state."""
    
    def __init__(self, seed_state=42):
        """Initialize cipher with a seed state.
        
        Args:
            seed_state: Initial 8-bit state (0-255)
        """
        self.state = seed_state & 0xFF  # Ensure 8-bit
        
    def lfsr_step(self, state):
        """Fibonacci LFSR state transition using taps at positions 7, 5, 4, 3.
        
        This creates a pseudo-random state update, but deterministic for a given state.
        
        Args:
            state: Current 8-bit state
            
        Returns:
            next_state: Updated 8-bit state
        """
        # Extract taps
        bit7 = (state >> 7) & 1
        bit5 = (state >> 5) & 1
        bit4 = (state >> 4) & 1
        bit3 = (state >> 3) & 1
        
        # XOR taps to create feedback bit
        feedback = bit7 ^ bit5 ^ bit4 ^ bit3
        
        # Shift left and insert feedback at bit 0
        next_state = ((state << 1) | feedback) & 0xFF
        
        return next_state
    
    def encrypt_nibble(self, plaintext, state):
        """Encrypt a single 4-bit nibble.
        
        Args:
            plaintext: 4-bit plaintext value (0-15)
            state: 8-bit current state (0-255)
            
        Returns:
            ciphertext: 4-bit encrypted value (0-15)
            next_state: 8-bit next state (0-255)
        """
        plaintext = plaintext & 0x0F  # Ensure 4-bit
        state = state & 0xFF  # Ensure 8-bit
        
        # Extract low 4 bits of state for mixing
        state_nibble = state & 0x0F
        
        # Encryption: plaintext XOR (state_nibble + state XOR)
        ciphertext = (plaintext ^ state_nibble ^ ((state >> 4) & 0x0F)) & 0x0F
        
        # State update: LFSR step followed by XOR with ciphertext feedback
        next_state = self.lfsr_step(state)
        next_state = (next_state ^ ciphertext) & 0xFF
        
        return ciphertext, next_state
    
    def decrypt_nibble(self, ciphertext, state):
        """Decrypt a single 4-bit nibble (reverse of encryption).
        
        Args:
            ciphertext: 4-bit encrypted value (0-15)
            state: 8-bit current state (0-255)
            
        Returns:
            plaintext: 4-bit decrypted value (0-15)
            next_state: 8-bit next state (0-255)
        """
        # Note: This uses the same state update as encrypt_nibble
        # So we cannot directly reverse without knowing ciphertext
        # For decryption to work, we need to work through the state transformation
        
        ciphertext = ciphertext & 0x0F
        state = state & 0xFF
        
        # Predict next state first (same as encrypt)
        next_state = self.lfsr_step(state)
        next_state = (next_state ^ ciphertext) & 0xFF
        
        # Recover plaintext: same XOR structure
        state_nibble = state & 0x0F
        plaintext = (ciphertext ^ state_nibble ^ ((state >> 4) & 0x0F)) & 0x0F
        
        return plaintext, next_state


def generate_state_table(cipher=None):
    """Generate complete state transition table for the cipher.
    
    Table structure:
    - Rows: 4,096 (256 states × 16 input values)
    - Columns: current_state, plaintext, ciphertext, next_state
    
    Args:
        cipher: SimpleCipher instance (creates new one if None)
        
    Returns:
        state_table: Pandas DataFrame with columns [state, plaintext, ciphertext, next_state]
    """
    if cipher is None:
        cipher = SimpleCipher()
    
    records = []
    
    # Iterate through all possible states and inputs
    for state in range(256):  # 0-255
        for plaintext in range(16):  # 0-15
            ciphertext, next_state = cipher.encrypt_nibble(plaintext, state)
            records.append({
                'state': state,
                'plaintext': plaintext,
                'ciphertext': ciphertext,
                'next_state': next_state
            })
    
    state_table = pd.DataFrame(records)
    return state_table


def bits_to_int(bits):
    """Convert list of bits to integer."""
    result = 0
    for bit in bits:
        result = (result << 1) | (1 if bit else 0)
    return result


def int_to_bits(value, num_bits):
    """Convert integer to list of bits."""
    bits = []
    for i in range(num_bits - 1, -1, -1):
        bits.append((value >> i) & 1)
    return bits


def normalize_for_nn(data, feature_type='io'):
    """Normalize data for neural network input/output.
    
    Converts integers to bit representations in [0, 1] range.
    
    Args:
        data: Integer value (0-255 for state, 0-15 for nibbles)
        feature_type: 'state' (8-bit), 'nibble' (4-bit), or 'io' (auto-detect)
        
    Returns:
        bits: List of floats in [0, 1] representing the bits
    """
    if feature_type == 'state':
        return [float(b) for b in int_to_bits(data, 8)]
    elif feature_type == 'nibble':
        return [float(b) for b in int_to_bits(data, 4)]
    else:
        # Auto-detect based on value range
        if data > 15:
            return [float(b) for b in int_to_bits(data & 0xFF, 8)]
        else:
            return [float(b) for b in int_to_bits(data & 0x0F, 4)]


def denormalize_from_nn(bits, feature_type='nibble'):
    """Denormalize neural network output bits back to integer.
    
    Args:
        bits: List of floats in [0, 1] representing bits (rounded to 0/1)
        feature_type: 'state' (8-bit) or 'nibble' (4-bit)
        
    Returns:
        value: Integer value (0-255 for state, 0-15 for nibble)
    """
    # Round to nearest 0 or 1
    binary_bits = [1 if b > 0.5 else 0 for b in bits]
    value = bits_to_int(binary_bits)
    
    if feature_type == 'state':
        return value & 0xFF
    elif feature_type == 'nibble':
        return value & 0x0F
    else:
        return value


def int_array_to_bits(values, num_bits):
    """Vectorized conversion of integer array to bit matrix.

    Args:
        values: 1D array of integers
        num_bits: Number of bits per value

    Returns:
        bits: 2D float array (len(values), num_bits)
    """
    values = np.asarray(values, dtype=np.int32)
    bit_positions = np.arange(num_bits - 1, -1, -1)
    return ((values[:, None] >> bit_positions) & 1).astype(np.float64)


def encode_state_table_for_nn(state_table):
    """Encode state table as normalized arrays for neural network training.

    Input format: [plaintext_bits (4) + state_bits (8)] = 12 input units
    Output format: [ciphertext_bits (4) + next_state_bits (8)] = 12 output units

    Args:
        state_table: Pandas DataFrame from generate_state_table()

    Returns:
        X: NumPy array of inputs (4096, 12)
        y: NumPy array of outputs (4096, 12)
    """
    pt_bits = int_array_to_bits(state_table['plaintext'].values, 4)
    st_bits = int_array_to_bits(state_table['state'].values, 8)
    ct_bits = int_array_to_bits(state_table['ciphertext'].values, 4)
    ns_bits = int_array_to_bits(state_table['next_state'].values, 8)

    X = np.hstack([pt_bits, st_bits])
    y = np.hstack([ct_bits, ns_bits])
    return X, y


def save_state_table(state_table, filepath):
    """Save state table to CSV."""
    state_table.to_csv(filepath, index=False)
    print(f"State table saved to {filepath}")


def load_state_table(filepath):
    """Load state table from CSV."""
    return pd.read_csv(filepath)


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("Simple 4-bit Stream Cipher - State Table Generation")
    print("=" * 60)
    
    # Create cipher and generate state table
    cipher = SimpleCipher(seed_state=42)
    state_table = generate_state_table(cipher)
    
    print(f"\nGenerated state table shape: {state_table.shape}")
    print(f"Expected: (4096, 4)")
    print("\nFirst 10 rows:")
    print(state_table.head(10))
    
    # Verify uniqueness of transitions
    unique_pairs = len(state_table[['state', 'plaintext']].drop_duplicates())
    print(f"\nUnique (state, plaintext) pairs: {unique_pairs}")
    print(f"Expected: 4096")
    
    # Show some example encryption
    print("\n" + "=" * 60)
    print("Example Encryptions")
    print("=" * 60)
    
    test_state = 42
    test_plaintexts = [0, 5, 10, 15]
    current_state = test_state
    
    for pt in test_plaintexts:
        ct, next_state = cipher.encrypt_nibble(pt, current_state)
        print(f"State: {current_state:3d} | Plaintext: {pt:2d} | Ciphertext: {ct:2d} | Next State: {next_state:3d}")
        current_state = next_state
    
    # Save state table
    output_file = 'state_table.csv'
    save_state_table(state_table, output_file)
    
    # Show neural network encoding
    X, y = encode_state_table_for_nn(state_table)
    print(f"\nEncoded for NN:")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Sample input: {X[0]}")
    print(f"Sample output: {y[0]}")
