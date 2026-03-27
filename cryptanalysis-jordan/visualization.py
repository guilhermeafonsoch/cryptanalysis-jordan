"""
Visualization module for cryptographic analysis project

Generates:
1. State table (CSV with interactive HTML viewer)
2. State diagrams (Graphviz DOT format, rendered as SVG)
3. Training curves (loss and accuracy over epochs)
4. Network architecture diagram (text-based showing Jordan structure)
"""

import numpy as np
import pandas as pd
import json
import os
from cipher_design import load_state_table


def generate_state_table_html(state_table, output_file='state_table.html', max_rows=100):
    """Generate interactive HTML viewer for state table.
    
    Args:
        state_table: Pandas DataFrame with state transitions
        output_file: Output HTML file
        max_rows: Show first N rows (full table is too large for browser)
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cipher State Table</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { padding: 10px; text-align: center; border: 1px solid #ddd; }
            th { background: #4CAF50; color: white; }
            tr:nth-child(even) { background: #f9f9f9; }
            tr:hover { background: #f0f0f0; }
            .stats { margin: 20px 0; padding: 10px; background: #e8f5e9; border-left: 4px solid #4CAF50; }
            input { padding: 8px; margin: 10px 0; }
            #stats { margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Cipher State Transition Table</h1>
            
            <div class="stats">
                <p><strong>Total transitions:</strong> 4,096 (256 states × 16 inputs)</p>
                <p><strong>Showing:</strong> First """ + str(max_rows) + """ rows for demonstration</p>
                <p><strong>Columns:</strong> Current State | Plaintext Input | Ciphertext Output | Next State</p>
            </div>
            
            <h2>Interactive State Search</h2>
            <label>Search by state (0-255):
                <input type="number" id="stateSearch" min="0" max="255" placeholder="Enter state...">
                <button onclick="searchState()">Search</button>
            </label>
            <div id="searchResults" style="margin-top: 15px;"></div>
            
            <h2>First """ + str(max_rows) + """ State Transitions</h2>
            <table>
                <tr>
                    <th>Current State</th>
                    <th>Plaintext Nibble (0-15)</th>
                    <th>Ciphertext Nibble (0-15)</th>
                    <th>Next State</th>
                </tr>
    """
    
    # Add first N rows
    for i, (_, row) in enumerate(state_table.head(max_rows).iterrows()):
        html_content += f"""
                <tr>
                    <td>{int(row['state'])}</td>
                    <td>{int(row['plaintext'])}</td>
                    <td>{int(row['ciphertext'])}</td>
                    <td>{int(row['next_state'])}</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>State Transition Explorer</h2>
            <p>Get all transitions from a specific state:</p>
            <label>State (0-255):
                <input type="number" id="stateExplore" min="0" max="255" placeholder="Enter state...">
                <button onclick="exploreState()">Explore</button>
            </label>
            <div id="exploreResults" style="margin-top: 15px; max-height: 400px; overflow-y: auto;"></div>
        </div>
        
        <script>
            const stateTableData = """ + state_table.to_json(orient='records') + """;
            
            function searchState() {
                const state = parseInt(document.getElementById('stateSearch').value);
                if (isNaN(state) || state < 0 || state > 255) {
                    alert('Please enter a state between 0 and 255');
                    return;
                }
                
                const matches = stateTableData.filter(row => row.state === state);
                let html = `<h3>Transitions from state ${state}: ${matches.length} results</h3><table style="width:100%; border-collapse:collapse;">`;
                html += '<tr style="background:#4CAF50;color:white;"><th>Plaintext</th><th>Ciphertext</th><th>Next State</th></tr>';
                
                matches.forEach(row => {
                    html += `<tr style="border:1px solid #ddd;padding:5px;"><td>${row.plaintext}</td><td>${row.ciphertext}</td><td>${row.next_state}</td></tr>`;
                });
                html += '</table>';
                
                document.getElementById('searchResults').innerHTML = html;
            }
            
            function exploreState() {
                const state = parseInt(document.getElementById('stateExplore').value);
                if (isNaN(state) || state < 0 || state > 255) {
                    alert('Please enter a state between 0 and 255');
                    return;
                }
                
                const matches = stateTableData.filter(row => row.state === state);
                let html = `<h3>All 16 transitions from state ${state}:</h3><div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;">`;
                
                matches.forEach(row => {
                    html += `<div style="border:1px solid #ddd;padding:10px;background:#f9f9f9;">
                        Input: ${row.plaintext} → Output: ${row.ciphertext}<br>
                        Next state: ${row.next_state}
                    </div>`;
                });
                html += '</div>';
                
                document.getElementById('exploreResults').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"State table HTML saved to {output_file}")


def generate_state_diagram_dot(state_table, output_file='state_diagram.dot', sample_states=10):
    """Generate Graphviz DOT file for state diagram (sample of states).
    
    Args:
        state_table: Pandas DataFrame
        output_file: Output .dot file
        sample_states: Number of states to include in diagram (full 256 is too large)
    """
    selected_states = list(range(0, 256, max(1, 256 // sample_states)))[:sample_states]
    
    dot_content = """digraph StateTransitions {
    rankdir=LR;
    node [shape=circle, style=filled, fillcolor=lightblue];
    edge [fontsize=8];
    
"""
    
    # Add nodes
    for state in selected_states:
        dot_content += f'    {state} [label="{state}"];\n'
    
    # Add sample transitions
    for state in selected_states:
        transitions = state_table[state_table['state'] == state].sample(min(2, len(state_table[state_table['state'] == state])))
        for _, row in transitions.iterrows():
            next_state = int(row['next_state'])
            if next_state in selected_states:
                plaintext = int(row['plaintext'])
                ciphertext = int(row['ciphertext'])
                dot_content += f'    {state} -> {next_state} [label="P:{plaintext}→C:{ciphertext}"];\n'
    
    dot_content += "}\n"
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(dot_content)
    print(f"State diagram DOT file saved to {output_file}")
    print(f"To render: graphviz state_diagram.dot -Tsvg -o state_diagram.svg")


def plot_training_curves(history_file, output_file='training_curves.html', name='Network'):
    """Generate HTML with Plotly training curves.
    
    Args:
        history_file: JSON file with training history
        output_file: Output HTML file
        name: Network name for title
    """
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Get the right data based on history structure
    if 'losses' in history:
        epochs = history.get('epochs', list(range(1, len(history['losses']) + 1)))
        losses = history['losses']
        label = 'Loss'
    else:
        epochs = history.get('epochs', [])
        losses = history.get('val_loss', history.get('train_loss', []))
        label = 'Validation Loss'
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{name} Training Curves</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 5px; }}
            h1 {{ color: #333; }}
            .chart-container {{ margin: 30px 0; }}
            .stats {{ padding: 10px; background: #e8f5e9; border-left: 4px solid #4CAF50; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{name} Training Curves</h1>
            
            <div class="stats">
                <p><strong>Total Epochs:</strong> {len(epochs)}</p>
                <p><strong>Final Loss:</strong> {losses[-1]:.6f}</p>
                <p><strong>Initial Loss:</strong> {losses[0]:.6f}</p>
            </div>
            
            <div class="chart-container">
                <h2>Training Loss Over Epochs</h2>
                <div id="loss_chart" style="width:100%;height:500px;"></div>
            </div>
        </div>
        
        <script>
            const epochs = {json.dumps(epochs)};
            const losses = {json.dumps(losses)};
            
            const trace = {{
                x: epochs,
                y: losses,
                type: 'scatter',
                mode: 'lines+markers',
                marker: {{size: 5, color: '#2196F3'}},
                line: {{width: 2}}
            }};
            
            const layout = {{
                title: '{name} - Loss Convergence',
                xaxis: {{title: 'Epoch'}},
                yaxis: {{title: '{label}'}},
                plot_bgcolor: '#f9f9f9',
                hovermode: 'closest'
            }};
            
            Plotly.newPlot('loss_chart', [trace], layout);
        </script>
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Training curves saved to {output_file}")


def generate_network_architecture_diagram(output_file='network_architecture.txt'):
    """Generate text-based Jordan network architecture diagram.
    
    Args:
        output_file: Output text file
    """
    diagram = """
════════════════════════════════════════════════════════════════════════════════
                    JORDAN NETWORK ARCHITECTURE
════════════════════════════════════════════════════════════════════════════════

ENCRYPTION NETWORK & CRYPTANALYSIS NETWORK (identical structure, different tasks)

Task 1 - Encryption:
    Input:  plaintext_nibble (4 bits) + state (8 bits) = 12 bits
    Output: ciphertext_nibble (4 bits) + next_state (8 bits) = 12 bits

Task 2 - Cryptanalysis:
    Input:  ciphertext_nibble (4 bits) + state (8 bits) = 12 bits
    Output: plaintext_nibble (4 bits) + next_state (8 bits) = 12 bits

───────────────────────────────────────────────────────────────────────────────

                            NETWORK STRUCTURE
                            
    ┌─────────────────────────────────────────────────────────────┐
    │                      CONTEXT LAYER (12 units)               │
    │                   [Feedback from Output]                    │
    └──────────────────────────┬──────────────────────────────────┘
                               │
                               │ (feedback loop)
                               ↓
    ┌──────────────────────────────────────────────────────────────┐
    │                  INPUT LAYER (24 units total)                │
    │  ┌────────────────────────┐      ┌─────────────────────┐    │
    │  │ Original Input (12)     │  +   │ Context (12)        │    │
    │  │ • 4 plaintext/cipher    │      │ • Previous output   │    │
    │  │ • 8 state bits          │      │   values            │    │
    │  └────────────────────────┘      └─────────────────────┘    │
    └──────────┬───────────────────────────────────────────────────┘
               │
               │ [24 × 10] weights + bias
               ↓
    ┌──────────────────────────────────────────────────────────────┐
    │            HIDDEN LAYER (10 units, sigmoid activation)       │
    │                                                               │
    │  ●─────────────────────────────────────────────────────●     │
    │  ●─────────────────────────────────────────────────────●     │
    │  ●─────────────────────────────────────────────────────●     │
    │  ●─────────────────────────────────────────────────────●     │
    │  ●─────────────────────────────────────────────────────●     │
    │  ●─────────────────────────────────────────────────────●     │
    │  ●─────────────────────────────────────────────────────●     │
    │  ●─────────────────────────────────────────────────────●     │
    │  ●─────────────────────────────────────────────────────●     │
    │  ●─────────────────────────────────────────────────────●     │
    └──────────┬───────────────────────────────────────────────────┘
               │
               │ [10 × 12] weights + bias
               ↓
    ┌──────────────────────────────────────────────────────────────┐
    │        OUTPUT LAYER (12 units, sigmoid activation)           │
    │  ┌─────────────────┐        ┌──────────────────┐             │
    │  │ Ciphertext (4)  │        │ Next State (8)   │             │
    │  │  or Plaintext   │        │                  │             │
    │  └─────────────────┘        └──────────────────┘             │
    └──────────┬───────────────────────────────────────────────────┘
               │
               │ (feedback loop → context layer)
               ↓
        [Output values stored as context]
        [Will be concatenated with next input]

───────────────────────────────────────────────────────────────────────────────

KEY FEATURES OF JORDAN NETWORK:

1. RECURRENCE: Output units feed back to input as "context"
   - Enables learning of sequential/stateful behavior
   - Each timestep sees current input + previous output
   - Simulates internal state in the network

2. PARAMETER COUNT:
   - W_input_hidden:    24 × 10 = 240 weights
   - W_hidden_output:   10 × 12 = 120 weights
   - Biases:            10 + 12 = 22
   - Total:             382 parameters

3. TRAINING:
   - Backpropagation with momentum
   - Context reset between samples (treat each transition independently)
   - Learning rate: 0.1
   - Momentum: 0.1

4. WHY JORDAN FOR CRYPTANALYSIS:
   - Cipher state is inherently sequential
   - Output (next state) becomes input context (current state)
   - Network learns state transitions implicitly through feedback
   - Better than feedforward for stateful transformations

═══════════════════════════════════════════════════════════════════════════════
"""
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(diagram)
    print(f"Network architecture diagram saved to {output_file}")


def main():
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Load data
    print("\nLoading state table...")
    state_table = load_state_table('state_table.csv')
    
    # Generate all visualizations
    print("\n1. Generating state table HTML...")
    generate_state_table_html(state_table, output_file='visualizations/state_table.html')
    
    print("\n2. Generating state diagram (Graphviz)...")
    generate_state_diagram_dot(state_table, output_file='visualizations/state_diagram.dot', sample_states=16)
    
    print("\n3. Generating training curves...")
    plot_training_curves('models/encryption_history.json', 
                           output_file='visualizations/encryption_training.html',
                           name='Encryption Network')
    plot_training_curves('models/cryptanalysis_history.json',
                           output_file='visualizations/cryptanalysis_training.html',
                           name='Cryptanalysis Network')
    
    print("\n4. Generating network architecture diagram...")
    generate_network_architecture_diagram(output_file='visualizations/network_architecture.txt')
    
    print("\n" + "=" * 60)
    print("All visualizations generated!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - visualizations/state_table.html           (Interactive state transitions)")
    print("  - visualizations/state_diagram.dot          (Graphviz state diagram)")
    print("  - visualizations/encryption_training.html   (Training curves)")
    print("  - visualizations/cryptanalysis_training.html(Training curves)")
    print("  - visualizations/network_architecture.txt   (ASCII diagram)")


if __name__ == '__main__':
    main()
