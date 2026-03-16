#!/usr/bin/env python3
"""Build master_kan_deepfake.ipynb by merging all 7 stage notebooks."""
import json, os

DIR = '/home/satvik/Desktop/dmml_project_KAN'
stages = [
    'stage1_phase_extraction.ipynb',
    'stage2_phase_analysis.ipynb',
    'stage3_pca_mlp_baseline.ipynb',
    'stage4_kan_phase.ipynb',
    'stage5_cnn_vit_baselines.ipynb',
    'stage6_ablations_robustness.ipynb',
    'stage7_evaluation_interpretability.ipynb',
]

# Collect all cells
all_cells = []

# Add master header
all_cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": [
        "# KAN-Driven Phase-Spectrum Analysis for Deepfake Detection\n",
        "## Master Notebook — All Stages (1-7)\n",
        "Run all cells sequentially on Kaggle with GPU T4.\n",
        "\n",
        "**Expected Runtime:** ~1.5-2 hours total\n",
        "\n",
        "| Stage | Description | Time |\n",
        "|-------|------------|------|\n",
        "| 1 | Phase Extraction | ~2 min |\n",
        "| 2 | Dataset Integration | ~5-8 min |\n",
        "| 3 | PCA & MLP Baseline | ~3-5 min |\n",
        "| 4 | KAN-Phase | ~5-8 min |\n",
        "| 5 | CNN & ViT Baselines | ~15-20 min |\n",
        "| 6 | Ablations & Robustness | ~40-60 min |\n",
        "| 7 | Evaluation & Interpretability | ~5-7 min |"
    ]
})

# Single install cell at top
all_cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": ["!pip install pykan kagglehub -q"]
})

# Track what we've seen to skip duplicate installs
seen_install = False

for stage_file in stages:
    path = os.path.join(DIR, stage_file)
    with open(path) as f:
        nb = json.load(f)
    
    stage_num = stage_file.split('_')[0].replace('stage', '')
    
    for cell in nb['cells']:
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Skip pip install cells (we have one master install at top)
        if cell['cell_type'] == 'code' and source.strip().startswith('!pip'):
            continue
        
        # Skip duplicate import+path cells for stages 3-7
        # (Stage 1 has the master imports, Stage 2 has dataset explorer)
        # For stages 3+, we need to skip the imports/path block but keep the logic
        if cell['cell_type'] == 'code' and int(stage_num) >= 3:
            # Check if this is a pure import+path cell
            has_kagglehub_download = 'kagglehub.dataset_download' in source
            if has_kagglehub_download:
                # Extract only the non-import, non-path lines
                lines = source.split('\n')
                keep_lines = []
                skip_patterns = [
                    'import ', 'from ', '%matplotlib', 'plt.rcParams',
                    'sns.set_style', 'INPUT_DIR', 'OUTPUT_DIR', 'CACHE_DIR',
                    'MODEL_DIR', 'ABL_DIR', 'RPT_DIR', 'ABLATION_DIR',
                    'os.makedirs', 'DEVICE =', 'kagglehub', 'print(f\'Device',
                    'print(f\'Dataset'
                ]
                for line in lines:
                    stripped = line.strip()
                    if not stripped or any(stripped.startswith(p) for p in skip_patterns):
                        continue
                    keep_lines.append(line)
                
                if keep_lines:
                    # Rebuild as a cell with just the non-import logic
                    cell = dict(cell)  # copy
                    cell['source'] = [l + '\n' for l in keep_lines]
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                    all_cells.append(cell)
                continue
        
        all_cells.append(cell)

# Build notebook
master = {
    "cells": all_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(DIR, 'master_kan_deepfake.ipynb')
with open(out_path, 'w') as f:
    json.dump(master, f, indent=1)

# Validate
import ast
code_cells = [c for c in all_cells if c['cell_type'] == 'code']
errors = 0
for i, cell in enumerate(code_cells):
    src = ''.join(cell['source'])
    clean = '\n'.join([l if not l.strip().startswith(('!','%')) else '# '+l for l in src.split('\n')])
    try:
        ast.parse(clean)
    except SyntaxError as e:
        print(f'  FAIL Cell {i}: {e.msg} at line {e.lineno}')
        errors += 1

total = len(all_cells)
md_cells = sum(1 for c in all_cells if c['cell_type'] == 'markdown')
print(f'Master notebook: {total} cells ({len(code_cells)} code, {md_cells} markdown)')
print(f'Syntax errors: {errors}')
print(f'Saved: {out_path}')
