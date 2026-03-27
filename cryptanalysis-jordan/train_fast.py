"""
Fast training script for Jordan networks using batch training without per-epoch evaluation.
This version trains quickly and saves models, then we evaluate separately.
"""

import numpy as np
import pandas as pd
import json
import time
from cipher_design import load_state_table, encode_state_table_for_nn, int_array_to_bits
from networks import JordanNetwork, save_network


def quick_train_network(network, X_train, y_train, epochs=50, batch_size=64, verbose=True):
    """Fast training without expensive per-epoch eval.
    
    Args:
        network: JordanNetwork to train
        X_train: Training inputs
        y_train: Training targets
        epochs: Number of epochs
        batch_size: Batch size for display
        verbose: Print progress
        
    Returns:
        history: Training metrics
    """
    n_train = X_train.shape[0]
    history = {'epochs': [], 'losses': []}
    
    if verbose:
        print(f"\nTraining network ({epochs} epochs, {n_train} samples, batch logging every {batch_size})...")
        print(f"{'Epoch':<8} {'Avg Loss':<12} {'Time':<8}")
        print("-" * 40)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_train)
        epoch_losses = []
        
        for batch_idx in range(0, n_train, batch_size):
            batch_indices = indices[batch_idx:batch_idx+batch_size]
            batch_loss = 0.0
            
            for idx in batch_indices:
                network.reset_context()
                output, hidden, hidden_input, output_input, x_with_context = network.forward(X_train[idx])
                loss = network.backward(X_train[idx], y_train[idx], output, hidden, hidden_input, output_input, x_with_context)
                batch_loss += loss
            
            epoch_losses.append(batch_loss / len(batch_indices))
        
        avg_loss = np.mean(epoch_losses)
        history['losses'].append(avg_loss)
        history['epochs'].append(epoch + 1)
        
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            elapsed = time.time() - start_time
            print(f"{epoch+1:<8} {avg_loss:<12.6f} {elapsed:>6.1f}s")
    
    if verbose:
        print("-" * 40)
        print(f"Training complete (final loss: {history['losses'][-1]:.6f})")
    
    return history


def main():
    print("=" * 60)
    print("Fast Training of Jordan Networks")
    print("=" * 60)
    
    # Load data
    print("\nLoading cipher state table...")
    state_table = load_state_table('state_table.csv')
    X, y = encode_state_table_for_nn(state_table)
    print(f"Data loaded: X shape={X.shape}, y shape={y.shape}")
    
    # Split train/val
    train_ratio = 0.8
    n_train = int(X.shape[0] * train_ratio)
    indices = np.random.permutation(X.shape[0])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
    
    # Train encryption network
    print("\n" + "=" * 60)
    print("ENCRYPTION NETWORK TRAINING")
    print("=" * 60)
    enc_net = JordanNetwork(input_size=12, hidden_size=10, output_size=12, learning_rate=0.1, momentum=0.1)
    enc_hist = quick_train_network(enc_net, X_train, y_train, epochs=50, batch_size=64)
    save_network(enc_net, 'models/encryption_network.pkl')
    
    # Train cryptanalysis network
    print("\n" + "=" * 60)
    print("CRYPTANALYSIS NETWORK TRAINING")
    print("=" * 60)
    
    # Build cryptanalysis dataset (vectorized)
    ct_bits = int_array_to_bits(state_table['ciphertext'].values, 4)
    st_bits = int_array_to_bits(state_table['state'].values, 8)
    pt_bits = int_array_to_bits(state_table['plaintext'].values, 4)
    ns_bits = int_array_to_bits(state_table['next_state'].values, 8)
    X_crypto = np.hstack([ct_bits, st_bits])
    y_crypto = np.hstack([pt_bits, ns_bits])
    
    train_idx2 = indices[:n_train]
    crypto_net = JordanNetwork(input_size=12, hidden_size=10, output_size=12, learning_rate=0.1, momentum=0.1)
    crypto_hist = quick_train_network(crypto_net, X_crypto[train_idx2], y_crypto[train_idx2], epochs=50, batch_size=64)
    save_network(crypto_net, 'models/cryptanalysis_network.pkl')
    
    # Save histories
    with open('models/encryption_history.json', 'w') as f:
        json.dump(enc_hist, f, indent=2)
    with open('models/cryptanalysis_history.json', 'w') as f:
        json.dump(crypto_hist, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete! Models saved to 'models/' directory")
    print("=" * 60)


if __name__ == '__main__':
    main()
