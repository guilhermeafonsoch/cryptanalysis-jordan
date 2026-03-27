# Cryptographic Analysis with Jordan Neural Networks

## Quick Start (30 seconds)

```bash
# Go to project directory
cd cryptanalysis-jordan

# Run all steps
python cipher_design.py      # Generate state table
python train_fast.py         # Train both networks
python visualization.py      # Create visualizations

# Open the dashboard
# Open visualizations/dashboard.html in your browser
```

## What This Does

✅ **Cipher Design** - Creates a 4-bit stream cipher with 4,096 state transitions  
✅ **Network Training** - Trains two Jordan networks to learn cipher mappings  
✅ **Visualizations** - Generates interactive dashboards and diagrams  
✅ **Dashboard** - Open `visualizations/dashboard.html` to explore everything  

## Files Generated

| File | Purpose |
|------|---------|
| `state_table.csv` | All 4,096 cipher state transitions |
| `models/*.pkl` | Trained neural networks |
| `models/*.json` | Training metrics (loss curves) |
| `visualizations/*.html` | Interactive visualizations |

## Main Dashboard

Open **`visualizations/dashboard.html`** to access:
- 📊 Project overview & statistics
- 🔒 Interactive encryption simulator
- 🔓 Interactive cryptanalysis simulator (break the cipher!)
- 🧠 Network architecture explanation
- 📈 Training curves and visualizations
- 📚 Links to all project files

## How It Works

```
Step 1: Cipher Design
  └─> Simple 4-bit XOR-based cipher with 8-bit state
      └─> Input: plaintext nibble (4 bits) + state (8 bits)
          Output: ciphertext nibble (4 bits) + next state (8 bits)

Step 2: Generate State Table
  └─> Create all 256 × 16 = 4,096 transition pairs
      └─> Used as training data for neural networks

Step 3: Train Networks
  └─> Encryption Network: learns (plaintext, state) → (ciphertext, next_state)
  └─> Cryptanalysis Network: learns (ciphertext, state) → (plaintext, next_state)

Step 4: Evaluate
  └─> Both networks achieve ~75-79% bit-level accuracy
      └─> Successfully learn the cipher transformation!
```

## Network Architecture

**Jordan Network** (Recurrent with context feedback):
```
Input (12 bits) + Context (12 bits from previous output)
        ↓
Hidden Layer (10 sigmoid units)
        ↓
Output (12 bits)
        ↓
Context feedback (connects back to input)
```

**Why Jordan?** The cipher is stateful - output becomes next input context!

## Key Results

| Metric | Encryption | Cryptanalysis |
|--------|-----------|---|
| Final Loss | 0.115 | 0.169 |
| Accuracy | 79% | 75% |
| Parameters | 382 | 382 |
| Training Time | ~8s | ~15s |

## Educational Value

This project teaches:
- ✅ How recurrent neural networks handle state
- ✅ The power of neural networks for learning mappings
- ✅ Cryptanalysis techniques using machine learning
- ✅ Why simple ciphers are vulnerable
- ✅ How to implement and visualize neural networks

## Files Reference

| File | What It Does |
|------|---|
| `cipher_design.py` | Implements 4-bit cipher, generates 4,096-row state table |
| `networks.py` | Jordan network class with forward/backward passes |
| `train_fast.py` | Training script (uses batch updates for speed) |
| `visualization.py` | Generates all visualizations automatically |
| `CRYPTANALYSIS.md` | Full technical documentation |

## Extending the Project

Want to try different configurations? Edit these:

```python
# In cipher_design.py
SimpleCipher.lfsr_step()  # Change feedback taps

# In networks.py
JordanNetwork(input_size=12, hidden_size=10, output_size=12)
              # Try: hidden_size=20, hidden_size=50

# In train_fast.py
epochs=50  # Try: 100, 200
learning_rate=0.1  # Try: 0.01, 0.05, 0.2
```

## Troubleshooting

**Q: Missing module error?**
```bash
pip install numpy pandas
```

**Q: Graphs not showing in HTML?**
- Make sure you have internet (uses CDN for Plotly)
- Or save the HTML offline with embedded libraries

**Q: Want to visualize the state diagram?**
```bash
# Install Graphviz
# Windows: choco install graphviz
# Mac: brew install graphviz
# Linux: apt install graphviz

# Then render the diagram
dot -Tsvg visualizations/state_diagram.dot -o visualizations/state_diagram.svg
```

## Fun Experiments

1. **Can you break the cipher?** Try the cryptanalysis simulator in the dashboard
2. **What happens with more hidden neurons?** Modify `networks.py` to use 20 or 50 hidden units
3. **Larger cipher?** Extend `cipher_design.py` to 8-bit inputs and 16-bit state
4. **Without context?** Compare Jordan vs feedforward networks
5. **What's the minimum accuracy needed to break it?** Verify cryptanalysis network accuracy

## Dashboard Navigation

```
📊 Overview
└─> Project statistics (4,096 transitions, 382 parameters)
└─> How Jordan networks work
└─> Educational value

🔒 Encryption
└─> Try encrypting any plaintext (0-15)
└─> See the cipher output

🔓 Cryptanalysis
└─> Try to recover plaintext from ciphertext!
└─> See attack success rate

🧠 Architecture
└─> Detailed network diagram
└─> Compare Jordan vs feedforward

📈 Visualizations
└─> Interactive state table explorer
└─> Training curves
└─> State transition diagrams
```

## Next Steps

1. ✅ Run `python cipher_design.py` to understand the cipher
2. ✅ Open `visualizations/dashboard.html` to see everything
3. ✅ Try the encryption & cryptanalysis simulators
4. ✅ Read `CRYPTANALYSIS.md` for deep technical details
5. ✅ Modify and extend the code!

---

**Happy learning!** 🧠🔐

For detailed technical information, see [CRYPTANALYSIS.md](CRYPTANALYSIS.md)
