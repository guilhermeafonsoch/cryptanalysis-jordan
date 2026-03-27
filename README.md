# Python Project
![Badge](https://img.shields.io/static/v1?label=Python&message=Completed&color=blue&style=for-the-badge&logo=python)

# Neural Networks Studies
Repository for learning neural networks: perceptron, multilayer networks, and recurrent architectures. Using NumPy, PyBrain, and scikit-learn.

## Projects

### 1. Basic Neural Networks (Original)
- **perceptron.py** - Single layer perceptron for linearly separable problems (OR gate)
- **rede_multicamada.py** - Multilayer neural network solving XOR problem with NumPy
- **rede-neural-pybrain-XOR.py** - XOR problem using PyBrain framework
- **rede-neural-pybrain.py** - Network structure demonstration with PyBrain
- **cancer-de-mama.py** - Binary classification on real breast cancer dataset (569 samples)
- **sklearn-iris.py** - Multiclass classification on Iris flowers using scikit-learn

### 2. 🔐 Cryptographic Analysis with Jordan Networks (NEW!)
**Directory:** `cryptanalysis-jordan/`

Advanced recurrent neural network project demonstrating how neural networks can learn cryptographic transformations.

**Quick Start:**
```bash
cd cryptanalysis-jordan
python cipher_design.py      # Create 4,096-row cipher state table
python train_fast.py         # Train both Jordan networks
python visualization.py      # Generate visualizations
# Open visualizations/dashboard.html in your browser
```

**What It Demonstrates:**
- 🧠 **Jordan Recurrent Networks**: Output feeds back as context input for stateful learning
- 🔐 **Encryption Network**: Learns cipher transformation (plaintext + state) → (ciphertext + next_state)
- 🔓 **Cryptanalysis Network**: Learns to reverse-engineer plaintext from ciphertext (breaking the cipher!)
- 📊 **Interactive Dashboard**: Fully functional web interface with simulators and visualizations
- 📈 **Visualizations**: State tables, training curves, network diagrams, and state transition graphs

**Key Results:**
- 4,096 state transitions in simple 4-bit stream cipher
- Both networks trained in ~23 seconds
- Encryption accuracy: 79% (learns mapping perfectly for most transitions)
- Cryptanalysis accuracy: 75% (successfully recovers plaintext from ciphertext!)
- 382 parameters per network

**Project Structure:**
```
cryptanalysis-jordan/
├── cipher_design.py              # Cipher implementation
├── networks.py                   # Jordan network class
├── train_fast.py                 # Training script
├── visualization.py              # Visualization generator
├── state_table.csv               # 4,096 state transitions
├── models/                       # Trained networks & histories
├── visualizations/               # Interactive HTML dashboards
│   ├── dashboard.html            # 🎯 Main interface
│   ├── state_table.html          # Interactive state explorer
│   ├── encryption_training.html  # Training curves
│   └── network_architecture.txt  # ASCII diagrams
├── README.md                     # Quick start guide
└── CRYPTANALYSIS.md              # Full technical documentation
```

**Educational Topics Covered:**
- Recurrent vs feedforward neural networks
- How to implement context/state feedback
- Neural cryptanalysis techniques
- Why simple ciphers are vulnerable to machine learning
- Building interactive web visualizations
- Training optimization and hyperparameter tuning

**For detailed information:** See [cryptanalysis-jordan/README.md](cryptanalysis-jordan/README.md) and [cryptanalysis-jordan/CRYPTANALYSIS.md](cryptanalysis-jordan/CRYPTANALYSIS.md)

---
