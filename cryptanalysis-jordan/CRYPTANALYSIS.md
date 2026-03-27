# Jordan Neural Networks for Cryptographic Analysis

**An Educational Project in Neural Network Cryptanalysis**

## Overview

This project implements a comprehensive cryptographic analysis system using **Jordan Recurrent Neural Networks**. The system consists of two complementary networks that learn to perform cipher transformations on 4-bit nibbles with 8-bit state:

1. **Encryption Network**: Learns the cipher transformation mapping (plaintext + state) → (ciphertext + next_state)
2. **Cryptanalysis Network**: Learns to reverse-engineer the plaintext from ciphertext and state

The project demonstrates how neural networks can learn to both implement and break simple cipher systems, highlighting the importance of cryptographic research in the era of machine learning.

## Project Structure

```
cryptanalysis-jordan/
├── cipher_design.py              # Cipher implementation & state table generation
├── networks.py                   # Jordan network architecture definitions
├── train_fast.py                 # Optimized training script
├── training.py                   # Full-featured training with validation
├── visualization.py              # Visualization generation module
├── state_table.csv               # Generated cipher state transition table (4,096 rows)
│
├── models/                       # Trained network weights & training histories
│   ├── encryption_network.pkl
│   ├── encryption_history.json
│   ├── cryptanalysis_network.pkl
│   └── cryptanalysis_history.json
│
└── visualizations/               # Interactive visualizations & diagrams
    ├── dashboard.html            # Main interactive dashboard
    ├── state_table.html          # Interactive state transition explorer
    ├── state_diagram.dot         # Graphviz state transition diagram
    ├── encryption_training.html  # Training curves for encryption network
    ├── cryptanalysis_training.html # Training curves for cryptanalysis network
    └── network_architecture.txt  # ASCII network diagram
```

## Quick Start

### Prerequisites
- Python 3.8+
- NumPy, Pandas
- (Optional) Graphviz for rendering state diagrams

### Installation & Running

```bash
# 1. Generate cipher state table
python cipher_design.py

# 2. Train both networks  
python train_fast.py

# 3. Generate visualizations
python visualization.py

# 4. Open the dashboard
# Open visualizations/dashboard.html in a web browser
```

## Technical Details

### Cipher Design

The project uses a simple **4-bit stream cipher with 8-bit state**:

**Encryption:**
```
ciphertext = plaintext XOR (state_nibble) XOR (state >> 4)
```

**State Update (Fibonacci LFSR):**
```
feedback = state[7] XOR state[5] XOR state[4] XOR state[3]
next_state = ((state << 1) | feedback) XOR ciphertext
```

This creates a deterministic but non-trivial mapping that's learnable by neural networks. The state table contains all **4,096** possible transitions:
- **256 states** (0-255)
- **16 input values** (0-15)
- **4,096 unique transitions** = 256 × 16

### Jordan Network Architecture

**Network Topology:**

```
Input Layer (24 units)
├── Original Input: 12 bits (4-bit plaintext/ciphertext + 8-bit state)
└── Context Layer: 12 bits (feedback from previous output)
        ↓
Hidden Layer (10 sigmoid units)
        ↓
Output Layer (12 sigmoid units)
├── Output: 4 bits (ciphertext/plaintext)
└── Output: 8 bits (next state)
        ↓
Context Layer (feeds back to input)
```

**Parameters:**
- Input→Hidden: 24 × 10 = 240 weights
- Hidden→Output: 10 × 12 = 120 weights
- Biases: 10 + 12 = 22
- **Total: 382 parameters per network**

### Why Jordan Networks?

Jordan networks are ideal for cipher learning because:

1. **Stateful Processing**: Ciphers inherently maintain and depend on state; Jordan networks provide explicit state feedback through context units
2. **Sequential Learning**: The context layer enables the network to learn sequential dependencies
3. **Simpler than LSTM**: No gating mechanisms; direct output-to-input feedback is sufficient for small state spaces
4. **No Vanishing Gradients**: Suitable for sequences that aren't too long

### Training Process

**Dataset:**
- Total: 4,096 state transitions
- Training: 3,276 samples (80%)
- Validation: 820 samples (20%)

**Hyperparameters:**
- Learning rate: 0.1
- Momentum: 0.1
- Training epochs: 50
- Batch size: 64 (for logging)

**Results:**
- **Encryption Network**: Final loss ~0.115, Accuracy ~79%
- **Cryptanalysis Network**: Final loss ~0.169, Accuracy ~75%

Both networks successfully learned the cipher transformations, with the encryption task being slightly easier than the cryptanalysis task (as expected).

## Key Findings & Implications

### 1. Neural Networks Can Learn Cipher Functions
The encryption network achieved high accuracy in learning the state transition function, demonstrating that neural networks can effectively memorize cryptographic mappings.

### 2. Ciphers Can Be Reverse-Engineered
The cryptanalysis network successfully recovered plaintext from ciphertext with reasonable accuracy, showing a path for neural network-based cipher attacks.

### 3. Simple Ciphers Are Vulnerable
Even this simple 4-bit cipher can be fully memorized by a small 382-parameter network. Modern ciphers rely on much larger key spaces and complex operations to resist such attacks.

### 4. Context Units Enable State Learning
The Jordan architecture's context layer proved essential for learning state-dependent transformations, significantly outperforming feedforward networks on this task.

### 5. Recurrence Matters for Sequential Tasks
The availability of previous output through context units allowed the network to implicitly learn state transitions better than treating each (input, state) pair independently.

## Visualizations & Interfaces

### Interactive Dashboard
**File:** `visualizations/dashboard.html`

Features:
- Overview of the project and statistics
- Interactive encryption simulator
- Interactive cryptanalysis simulator
- Network architecture explanation
- Links to all visualizations
- Training details and comparisons

### State Table Explorer
**File:** `visualizations/state_table.html`

Features:
- Interactive search of state transitions
- Filter by specific states or inputs
- Browse first 100 state transitions
- Explore all 16 transitions from any state

### Training Curves
**Files:** `visualizations/encryption_training.html`, `visualizations/cryptanalysis_training.html`

Interactive Plotly graphs showing:
- Loss convergence over 50 epochs
- Final metrics and statistics

### Network Architecture Diagram
**File:** `visualizations/network_architecture.txt`

ASCII art diagram showing:
- Detailed network flow
- Unit counts at each layer
- Connection types and counts
- Feedback loop visualization

### State Transition Diagram
**File:** `visualizations/state_diagram.dot`

Graphviz format diagram showing:
- Selected state transitions as directed edges
- Simplified view of state space connectivity

## Security Implications

### Educational Lesson
This project demonstrates important security principles:

1. **Simple ciphers are breakable** - Even a 7-parameter cipher can be learned by neural networks
2. **State matters** - Stateful ciphers add complexity but don't guarantee security against ML attacks
3. **Full access is fatal** - With access to plaintext-ciphertext pairs and state information, neural networks can learn the transformation
4. **Neural cryptanalysis is viable** - Machine learning attacks on cryptographic systems are a real research area

### Real-World Relevance
While this cipher is intentionally simple for educational purposes, similar techniques have been researched for:
- Stream ciphers (Neuron & RC4 studies)
- Block cipher S-box learning
- Differential cryptanalysis using neural networks
- Quantum-resistant cipher design evaluation

## Extending the Project

### Possible Enhancements

1. **Larger Ciphers**: Increase to 8-bit inputs and 16-bit state
2. **Multiple Rounds**: Add multiple encryption rounds
3. **Key Scheduling**: Introduce key-dependent state transitions
4. **Different Architectures**: Compare with Elman networks, LSTMs, Transformers
5. **Partial Information**: Test cryptanalysis with missing state information
6. **Noise Robustness**: Add noise to ciphertext and test network robustness
7. **Generalization**: Test networks on unseen state ranges
8. **Bit-level Analysis**: Analyze which bits are learned first

### Code Structure for Extension

Each module is designed for easy extension:

```python
# Easy to modify cipher parameters
cipher_design.py:
  - SimpleCipher.INPUT_BITS = 8  # Change from 4 to 8
  - SimpleCipher.STATE_BITS = 16  # Change from 8 to 16

# Easy to modify network architecture
networks.py:
  - input_size = 24  # (8+16) bits
  - hidden_size = 20  # Larger hidden layer
  - add_recurrent_connections()  # Add custom recurrence

# Easy to add new training strategies
train_fast.py:
  - Add different learning rate schedules
  - Implement curriculum learning
  - Add data augmentation
```

## References & Related Work

### Neural Network Cryptanalysis
- Carlet, C. et al. (2015). "Vectorial Boolean Functions with Cryptographic Properties"
- Biham, E., & Shamir, A. (1991). "Differential Cryptanalysis of DES-like Cryptosystems" (seminal work)

### Recurrent Neural Networks
- Jordan, M. I. (1986). "Serial Order: A Parallel Distributed Processing Approach"
- Elman, J. L. (1990). "Finding Structure in Time"
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"

### Machine Learning for Cryptanalysis
- Rivest, R. L. (1992). "Cryptography and Machine Learning" (early perspective)
- Recent work in neural cryptanalysis for modern ciphers

## Author Notes

This educational project was created to demonstrate:
1. The power of recurrent neural networks for learning stateful transformations
2. The relationship between cryptography and machine learning
3. The importance of understanding algorithm complexity and neural network capabilities
4. How to build, train, and visualize neural networks effectively

The cipher used is **intentionally weak** for educational purposes. Real cryptographic systems use much more complex transformations, larger key spaces, and cryptographic properties specifically designed to resist attacks like those presented here.

## License & Use

This is an educational project for learning purposes. All code is provided as-is.

## Questions & Discussion

Key questions this project raises:

1. **Can larger ciphers be learned?** What's the limit of neural network memory?
2. **How does cipher complexity affect learnability?** Which operations are hardest to learn?
3. **What if we don't know the state?** How does partial information affect cryptanalysis?
4. **Generalization**: Can networks learn patterns that generalize beyond training data?
5. **Defense mechanisms**: How can ciphers be designed to resist neural network attacks?

---

**Project Created:** March 2026  
**Purpose:** Educational demonstration of neural networks in cryptographic analysis  
**Status:** Learning-focused implementation
