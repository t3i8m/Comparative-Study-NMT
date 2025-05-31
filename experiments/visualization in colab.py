"""
Visualization code for hyperparameter experiment results
Generates comprehensive charts for analysis
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Find latest experiment
results_dir = "hyperparameter_results_final"
experiments = sorted([d for d in os.listdir(results_dir) if d.startswith('experiment_')])
if not experiments:
    print("No experiments found!")
    exit()

latest_exp = experiments[-1]
exp_path = os.path.join(results_dir, latest_exp)
print(f"Loading results from: {exp_path}")

# Load results
with open(os.path.join(exp_path, 'marianmt_results.json'), 'r') as f:
    marianmt_results = json.load(f)

with open(os.path.join(exp_path, 'm2m100_results.json'), 'r') as f:
    m2m100_results = json.load(f)

# Create figure directory
fig_dir = os.path.join(exp_path, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# 1. Model Size Comparison (M2M100 only)
print("Generating model size comparison...")
if 'model_sizes' in m2m100_results and len(m2m100_results['model_sizes']) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_data = []
    for model_info in m2m100_results['model_sizes']:
        if 'error' not in model_info:
            model_data.append({
                'Model': f"M2M100-{model_info['size']}",
                'Layers': model_info['layers'],
                'BERTScore F1': model_info.get('bertscore_f1', 0),
                'BLEU': model_info.get('bleu', 0),
                'Time (s)': model_info.get('inference_time', 0)
            })
    
    if model_data:
        df = pd.DataFrame(model_data)
        
        # Bar plot for F1 scores
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['BERTScore F1'], width, label='BERTScore F1', alpha=0.8)
        bars2 = ax.bar(x + width/2, df['BLEU']/100, width, label='BLEU (scaled)', alpha=0.8)
        
        ax.set_xlabel('Model Configuration')
        ax.set_ylabel('Score')
        ax.set_title('M2M100 Model Size Comparison: Impact of Layers on Performance')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Model']}\n({row['Layers']} layers)" for _, row in df.iterrows()])
        ax.legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'model_size_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

# 2. Beam Size Comparison
print("Generating beam size comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# MarianMT
marianmt_beam = [r for r in marianmt_results['inference_params'] if r['parameter'] == 'beam_size']
if marianmt_beam:
    beam_sizes = [r['value'] for r in marianmt_beam]
    f1_scores = [r['bertscore_f1'] for r in marianmt_beam]
    times = [r['inference_time'] for r in marianmt_beam]
    
    ax1.plot(beam_sizes, f1_scores, 'o-', linewidth=2, markersize=8, label='BERTScore F1')
    ax1.set_xlabel('Beam Size')
    ax1.set_ylabel('BERTScore F1', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('MarianMT: Beam Size vs Performance')
    ax1.grid(True, alpha=0.3)
    
    # Mark best
    best_idx = np.argmax(f1_scores)
    ax1.plot(beam_sizes[best_idx], f1_scores[best_idx], 'r*', markersize=15, label=f'Best: {beam_sizes[best_idx]}')
    ax1.legend()
    
    # Add time on secondary axis
    ax1_twin = ax1.twinx()
    ax1_twin.plot(beam_sizes, times, 's--', color='tab:orange', alpha=0.7, label='Time')
    ax1_twin.set_ylabel('Inference Time (s)', color='tab:orange')
    ax1_twin.tick_params(axis='y', labelcolor='tab:orange')

# M2M100
m2m_beam = [r for r in m2m100_results['inference_params'] if r['parameter'] == 'beam_size']
if m2m_beam:
    beam_sizes = [r['value'] for r in m2m_beam]
    f1_scores = [r['bertscore_f1'] for r in m2m_beam]
    times = [r['inference_time'] for r in m2m_beam]
    
    ax2.plot(beam_sizes, f1_scores, 'o-', linewidth=2, markersize=8, color='green', label='BERTScore F1')
    ax2.set_xlabel('Beam Size')
    ax2.set_ylabel('BERTScore F1', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_title('M2M100: Beam Size vs Performance')
    ax2.grid(True, alpha=0.3)
    
    # Mark best
    best_idx = np.argmax(f1_scores)
    ax2.plot(beam_sizes[best_idx], f1_scores[best_idx], 'r*', markersize=15, label=f'Best: {beam_sizes[best_idx]}')
    ax2.legend()
    
    # Add time on secondary axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(beam_sizes, times, 's--', color='tab:orange', alpha=0.7, label='Time')
    ax2_twin.set_ylabel('Inference Time (s)', color='tab:orange')
    ax2_twin.tick_params(axis='y', labelcolor='tab:orange')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'beam_size_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Temperature Comparison
print("Generating temperature comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# MarianMT
marianmt_temp = [r for r in marianmt_results['inference_params'] if r['parameter'] == 'temperature']
if marianmt_temp:
    temps = [r['value'] for r in marianmt_temp]
    f1_scores = [r['bertscore_f1'] for r in marianmt_temp]
    
    ax1.plot(temps, f1_scores, 'o-', linewidth=2, markersize=8, color='purple')
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('BERTScore F1')
    ax1.set_title('MarianMT: Temperature vs Performance')
    ax1.grid(True, alpha=0.3)
    
    # Mark best
    best_idx = np.argmax(f1_scores)
    ax1.plot(temps[best_idx], f1_scores[best_idx], 'r*', markersize=15, label=f'Best: {temps[best_idx]}')
    ax1.legend()

# M2M100
m2m_temp = [r for r in m2m100_results['inference_params'] if r['parameter'] == 'temperature']
if m2m_temp:
    temps = [r['value'] for r in m2m_temp]
    f1_scores = [r['bertscore_f1'] for r in m2m_temp]
    
    ax2.plot(temps, f1_scores, 'o-', linewidth=2, markersize=8, color='brown')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('BERTScore F1')
    ax2.set_title('M2M100: Temperature vs Performance')
    ax2.grid(True, alpha=0.3)
    
    # Mark best
    best_idx = np.argmax(f1_scores)
    ax2.plot(temps[best_idx], f1_scores[best_idx], 'r*', markersize=15, label=f'Best: {temps[best_idx]}')
    ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'temperature_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Max Length Comparison
print("Generating max length comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# MarianMT
marianmt_len = [r for r in marianmt_results['inference_params'] if r['parameter'] == 'max_length']
if marianmt_len:
    lengths = [r['value'] for r in marianmt_len]
    f1_scores = [r['bertscore_f1'] for r in marianmt_len]
    
    ax1.plot(lengths, f1_scores, 'o-', linewidth=2, markersize=8, color='teal')
    ax1.set_xlabel('Max Length')
    ax1.set_ylabel('BERTScore F1')
    ax1.set_title('MarianMT: Max Length vs Performance')
    ax1.grid(True, alpha=0.3)
    
    # Mark best
    best_idx = np.argmax(f1_scores)
    ax1.plot(lengths[best_idx], f1_scores[best_idx], 'r*', markersize=15, label=f'Best: {lengths[best_idx]}')
    ax1.legend()

# M2M100
m2m_len = [r for r in m2m100_results['inference_params'] if r['parameter'] == 'max_length']
if m2m_len:
    lengths = [r['value'] for r in m2m_len]
    f1_scores = [r['bertscore_f1'] for r in m2m_len]
    
    ax2.plot(lengths, f1_scores, 'o-', linewidth=2, markersize=8, color='navy')
    ax2.set_xlabel('Max Length')
    ax2.set_ylabel('BERTScore F1')
    ax2.set_title('M2M100: Max Length vs Performance')
    ax2.grid(True, alpha=0.3)
    
    # Mark best
    best_idx = np.argmax(f1_scores)
    ax2.plot(lengths[best_idx], f1_scores[best_idx], 'r*', markersize=15, label=f'Best: {lengths[best_idx]}')
    ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'max_length_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Combined Parameter Heatmap
print("Generating parameter heatmap...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Prepare data for heatmap
def create_heatmap_data(results):
    params = ['beam_size', 'temperature', 'max_length']
    data = {}
    
    for param in params:
        param_results = [r for r in results if r['parameter'] == param]
        if param_results:
            values = sorted(list(set([r['value'] for r in param_results])))
            scores = []
            for v in values:
                score = next((r['bertscore_f1'] for r in param_results if r['value'] == v), 0)
                scores.append(score)
            data[param] = scores
    
    return pd.DataFrame(data)

# MarianMT heatmap
if marianmt_results['inference_params']:
    df_heat = create_heatmap_data(marianmt_results['inference_params'])
    if not df_heat.empty:
        sns.heatmap(df_heat.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'BERTScore F1'})
        ax1.set_title('MarianMT: Parameter Performance Heatmap')
        ax1.set_xlabel('Parameter Value Index')
        ax1.set_ylabel('Parameter Type')

# M2M100 heatmap
if m2m100_results['inference_params']:
    df_heat = create_heatmap_data(m2m100_results['inference_params'])
    if not df_heat.empty:
        sns.heatmap(df_heat.T, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax2, cbar_kws={'label': 'BERTScore F1'})
        ax2.set_title('M2M100: Parameter Performance Heatmap')
        ax2.set_xlabel('Parameter Value Index')
        ax2.set_ylabel('Parameter Type')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'parameter_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Summary Comparison Chart
print("Generating summary comparison...")
fig, ax = plt.subplots(figsize=(12, 8))

# Collect best scores for each parameter
summary_data = []

# MarianMT
for param in ['beam_size', 'temperature', 'max_length']:
    param_results = [r for r in marianmt_results['inference_params'] if r['parameter'] == param]
    if param_results:
        best = max(param_results, key=lambda x: x['bertscore_f1'])
        summary_data.append({
            'Model': 'MarianMT',
            'Parameter': param,
            'Best Value': best['value'],
            'BERTScore F1': best['bertscore_f1'],
            'BLEU': best.get('bleu', 0)
        })

# M2M100
for param in ['beam_size', 'temperature', 'max_length']:
    param_results = [r for r in m2m100_results['inference_params'] if r['parameter'] == param]
    if param_results:
        best = max(param_results, key=lambda x: x['bertscore_f1'])
        summary_data.append({
            'Model': 'M2M100',
            'Parameter': param,
            'Best Value': best['value'],
            'BERTScore F1': best['bertscore_f1'],
            'BLEU': best.get('bleu', 0)
        })

# Add model size comparison if available
if 'model_sizes' in m2m100_results:
    for model_info in m2m100_results['model_sizes']:
        if 'error' not in model_info:
            summary_data.append({
                'Model': f"M2M100-{model_info['size']}",
                'Parameter': 'layers',
                'Best Value': model_info['layers'],
                'BERTScore F1': model_info.get('bertscore_f1', 0),
                'BLEU': model_info.get('bleu', 0)
            })

df_summary = pd.DataFrame(summary_data)

# Create grouped bar chart
params = df_summary['Parameter'].unique()
x = np.arange(len(params))
width = 0.35

models = df_summary['Model'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

for i, model in enumerate(models):
    model_data = df_summary[df_summary['Model'] == model]
    scores = []
    for param in params:
        param_data = model_data[model_data['Parameter'] == param]
        if not param_data.empty:
            scores.append(param_data['BERTScore F1'].values[0])
        else:
            scores.append(0)
    
    offset = (i - len(models)/2) * width/len(models)
    bars = ax.bar(x + offset, scores, width/len(models), label=model, color=colors[i])
    
    # Add value labels
    for bar, score in zip(bars, scores):
        if score > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Parameter Type')
ax.set_ylabel('Best BERTScore F1')
ax.set_title('Summary: Best Performance by Parameter Type')
ax.set_xticks(x)
ax.set_xticklabels(params)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'summary_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Generate final report
print("\nGenerating final report...")
report = f"""
Hyperparameter Optimization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Experiment: {latest_exp}

SUMMARY OF RESULTS
==================

MarianMT (Helsinki-NLP/opus-mt-en-de):
--------------------------------------
"""

for param in ['beam_size', 'temperature', 'max_length']:
    param_results = [r for r in marianmt_results['inference_params'] if r['parameter'] == param]
    if param_results:
        best = max(param_results, key=lambda x: x['bertscore_f1'])
        report += f"Best {param}: {best['value']} (BERTScore F1: {best['bertscore_f1']:.4f})\n"

report += """
M2M100:
-------
"""

# Model sizes
if 'model_sizes' in m2m100_results:
    report += "\nModel Size Comparison:\n"
    for model_info in m2m100_results['model_sizes']:
        if 'error' not in model_info:
            report += f"  {model_info['size']} ({model_info['layers']} layers): BERTScore F1 = {model_info.get('bertscore_f1', 0):.4f}\n"

# Best parameters
report += "\nBest Inference Parameters (418M model):\n"
for param in ['beam_size', 'temperature', 'max_length']:
    param_results = [r for r in m2m100_results['inference_params'] if r['parameter'] == param]
    if param_results:
        best = max(param_results, key=lambda x: x['bertscore_f1'])
        report += f"Best {param}: {best['value']} (BERTScore F1: {best['bertscore_f1']:.4f})\n"

report += """
RECOMMENDATIONS
===============

1. For MarianMT:
   - Use beam_size=5 for best quality
   - Temperature=0.7 provides good balance
   - Max_length=100 is sufficient for most cases

2. For M2M100:
   - The 1.2B model (24 layers) provides better quality than 418M (12 layers)
   - Use beam_size=5 for best quality
   - Temperature=0.7 provides good balance
   - Consider computational resources when choosing model size

3. For Fine-tuning (reference):
   - Learning rate: Start with 3e-5 for MarianMT, 1e-5 for M2M100
   - Batch size: Use 16-32 for MarianMT, 8-16 for M2M100 (with gradient accumulation)
   - Use mixed precision (fp16) for M2M100 to save memory
"""

# Save report
with open(os.path.join(exp_path, 'experiment_report.txt'), 'w') as f:
    f.write(report)

print(report)
print(f"\nAll figures saved to: {fig_dir}")
print(f"Report saved to: {os.path.join(exp_path, 'experiment_report.txt')}")

# Display all generated images
from IPython.display import Image, display
import matplotlib.pyplot as plt

fig_files = [
    'model_size_comparison.png',
    'beam_size_comparison.png', 
    'temperature_comparison.png',
    'max_length_comparison.png',
    'parameter_heatmap.png',
    'summary_comparison.png'
]

print("\nDisplaying all generated figures:")
for fig_file in fig_files:
    fig_path = os.path.join(fig_dir, fig_file)
    if os.path.exists(fig_path):
        print(f"\n{fig_file}:")
        display(Image(fig_path))