"""
Jordan Neural Networks for Encryption and Cryptanalysis

This module implements two recurrent neural networks based on the Jordan architecture:
1. Encryption Network: Learns the cipher transformation (plaintext + state → ciphertext + next_state)
2. Cryptanalysis Network: Learns to reverse-engineer plaintext (ciphertext + state → plaintext + next_state)

Architecture Details:
- Input Layer: 12 units + 12 context units (4-bit plaintext/ciphertext + 8-bit state + feedback from output)
- Hidden Layer: 10 units with sigmoid activation
- Output Layer: 12 units (4-bit ciphertext/plaintext + 8-bit next_state)
- Context Layer: Feedback from output units to input (Jordan architecture feature)

This module implements a simplified recurrent architecture using NumPy, following the patterns
from the existing neural networks repository (rede_multicamada.py, cancer-de-mama.py).
The Jordan architecture is realized by feeding output values back to input layer for the next timestep.
"""

import numpy as np
import pickle
import os


def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    return x * (1.0 - x)


class JordanNetwork:
    """Simple Jordan Recurrent Neural Network using NumPy.
    
    This is a simplified implementation of a Jordan network suitable for learning
    the cipher transformation. The Jordan architecture is characterized by feedback
    connections from output units back to a context layer at the input.
    
    Architecture:
    - Input layer: input_size units + context_size units (feedback from previous output)
    - Hidden layer: hidden_size units with sigmoid activation
    - Output layer: output_size units with sigmoid activation
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, momentum=0.1):
        """Initialize Jordan network.
        
        Args:
            input_size: Size of input (e.g., 12 for 4-bit plaintext + 8-bit state)
            hidden_size: Size of hidden layer (e.g., 10)
            output_size: Size of output layer (e.g., 12 for 4-bit ciphertext + 8-bit next_state)
            learning_rate: Learning rate for weight updates
            momentum: Momentum factor for smoothing weight updates
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Context layer = same size as output (feedback from output to input)
        self.context_size = output_size
        self.total_input_size = input_size + self.context_size
        
        # Initialize weights with small random values
        self.W_input_hidden = np.random.randn(self.total_input_size, hidden_size) * 0.1
        self.W_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        
        # Initialize biases
        self.b_hidden = np.zeros(hidden_size)
        self.b_output = np.zeros(output_size)
        
        # Initialize momentum matrices
        self.dW_input_hidden = np.zeros_like(self.W_input_hidden)
        self.dW_hidden_output = np.zeros_like(self.W_hidden_output)
        self.db_hidden = np.zeros_like(self.b_hidden)
        self.db_output = np.zeros_like(self.b_output)
        
        # Context layer state (initialized to zeros, will update with output feedback)
        self.context = np.zeros(self.context_size)
        
    def forward(self, x, context=None):
        """Forward pass with context feedback.
        
        Args:
            x: Input vector (input_size,)
            context: Context vector from previous output (context_size,). If None, use stored context.
            
        Returns:
            output: Output vector (output_size,)
            hidden: Hidden layer activations for backprop
        """
        if context is None:
            context = self.context
        
        # Concatenate input with context (Jordan architecture)
        x_with_context = np.concatenate([x, context])
        
        # Hidden layer
        hidden_input = np.dot(x_with_context, self.W_input_hidden) + self.b_hidden
        hidden = sigmoid(hidden_input)
        
        # Output layer
        output_input = np.dot(hidden, self.W_hidden_output) + self.b_output
        output = sigmoid(output_input)
        
        # Update context for next timestep
        self.context = output.copy()
        
        return output, hidden, hidden_input, output_input, x_with_context
    
    def backward(self, x, y, output, hidden, hidden_input, output_input, x_with_context):
        """Backward pass (backpropagation through time, simplified for single timestep).
        
        Args:
            x: Input vector
            y: Target output vector
            output: Network output from forward pass
            hidden: Hidden layer activations from forward pass
            hidden_input: Hidden layer pre-activation from forward pass
            output_input: Output layer pre-activation from forward pass
            x_with_context: Concatenated input with context
            
        Returns:
            loss: Mean squared error for this sample
        """
        # Output layer error
        output_error = output - y
        loss = np.mean(output_error ** 2)

        # Output layer gradient
        output_delta = output_error * sigmoid_derivative(output)

        # Hidden layer gradient
        hidden_error = np.dot(output_delta, self.W_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden)

        # Weight gradients
        dW_hidden_output = np.outer(hidden, output_delta)
        dW_input_hidden = np.outer(x_with_context, hidden_delta)
        db_output = output_delta
        db_hidden = hidden_delta
        
        # Update weights with momentum
        self.dW_hidden_output = self.momentum * self.dW_hidden_output + dW_hidden_output
        self.dW_input_hidden = self.momentum * self.dW_input_hidden + dW_input_hidden
        self.db_output = self.momentum * self.db_output + db_output
        self.db_hidden = self.momentum * self.db_hidden + db_hidden
        
        # Apply updates
        self.W_hidden_output -= self.learning_rate * self.dW_hidden_output
        self.W_input_hidden -= self.learning_rate * self.dW_input_hidden
        self.b_output -= self.learning_rate * self.db_output
        self.b_hidden -= self.learning_rate * self.db_hidden
        
        return loss
    
    def reset_context(self):
        """Reset context layer to zeros (for new sequence)."""
        self.context = np.zeros(self.context_size)
    
    def get_info(self):
        """Get network information."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'context_size': self.context_size,
            'total_input_size': self.total_input_size,
            'num_parameters': (
                self.W_input_hidden.size + self.W_hidden_output.size +
                self.b_hidden.size + self.b_output.size
            ),
            'learning_rate': self.learning_rate,
            'momentum': self.momentum
        }


def save_network(network, filepath):
    """Save network weights and structure to file.
    
    Args:
        network: JordanNetwork instance
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(network, f)
    print(f"Network saved to {filepath}")


def load_network(filepath):
    """Load network from file.
    
    Args:
        filepath: Path to saved network file
        
    Returns:
        network: Loaded JordanNetwork instance
    """
    with open(filepath, 'rb') as f:
        network = pickle.load(f)
    print(f"Network loaded from {filepath}")
    return network


if __name__ == '__main__':
    print("=" * 60)
    print("Jordan Networks for Cryptographic Analysis")
    print("=" * 60)
    
    # Create encryption network
    print("\nCreating Encryption Network...")
    enc_network = JordanNetwork(input_size=12, hidden_size=10, output_size=12, 
                                learning_rate=0.1, momentum=0.1)
    enc_info = enc_network.get_info()
    print(f"Encryption Network Info:")
    print(f"  - Input size: {enc_info['input_size']}")
    print(f"  - Hidden size: {enc_info['hidden_size']}")
    print(f"  - Output size: {enc_info['output_size']}")
    print(f"  - Total input (with context): {enc_info['total_input_size']}")
    print(f"  - Parameters: {enc_info['num_parameters']}")
    
    # Create cryptanalysis network
    print("\nCreating Cryptanalysis Network...")
    crypto_network = JordanNetwork(input_size=12, hidden_size=10, output_size=12,
                                   learning_rate=0.1, momentum=0.1)
    crypto_info = crypto_network.get_info()
    print(f"Cryptanalysis Network Info:")
    print(f"  - Input size: {crypto_info['input_size']}")
    print(f"  - Hidden size: {crypto_info['hidden_size']}")
    print(f"  - Output size: {crypto_info['output_size']}")
    print(f"  - Total input (with context): {crypto_info['total_input_size']}")
    print(f"  - Parameters: {crypto_info['num_parameters']}")
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    test_input = np.random.rand(12)
    print(f"Test input shape: {test_input.shape}")
    
    output, hidden, _, _, _ = enc_network.forward(test_input)
    print(f"Encryption network output shape: {output.shape}")
    print(f"Encryption network hidden shape: {hidden.shape}")
    
    enc_network.reset_context()
    output2, hidden2, _, _, _ = enc_network.forward(test_input)
    print(f"After reset, output == first output: {np.allclose(output, output2)}")
    
    print("\n" + "=" * 60)
    print("Networks ready for training")
    print("=" * 60)
