# Explainable AI-Driven Beamforming for Adaptive MIMO Systems in 5G

This repository contains the implementation of an XAI-driven beamforming optimization system for 5G MIMO communications. The project focuses on using explainable AI techniques to enhance beamforming performance in static user scenarios.

## Project Overview

### Objectives
- Generate realistic 5G MIMO channel datasets using Sionna
- Implement deep learning-based beamforming optimization
- Apply XAI techniques to understand and explain beamforming decisions
- Evaluate performance against traditional beamforming methods

### Key Features
- 4x4 MIMO system simulation with Nvidia Sionna library
- TDL-A channel model implementation
- Deep learning-based beamforming optimization
- XAI analysis and visualization
- Comprehensive performance evaluation

## Repository Structure
```
.
├── data/
│   ├── training/
│   ├── validation/
│   └── test/
├── src/
│   ├── config.py              # Configuration parameters
│   ├── channel_generator.py   # Channel dataset generation
│   ├── channel_analyzer.py    # Channel analysis tools
│   ├── models/
│   │   ├── beamformer.py     # Deep learning model
│   │   └── baseline.py       # Traditional beamforming
│   ├── xai/
│   │   ├── shap_analysis.py  # SHAP value computation
│   │   └── visualizer.py     # XAI visualization tools
│   └── training/
│       ├── trainer.py        # Model training
│       └── evaluator.py      # Performance evaluation
└── notebooks/
    ├── dataset_exploration.ipynb
    ├── model_training.ipynb
    └── xai_analysis.ipynb
```

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Sionna
- SHAP
- NumPy
- Matplotlib
- H5py

```bash
pip install -r requirements.txt
```

## Dataset Generation
The channel dataset is generated using Sionna's TDL-A model with the following parameters:
- 4x4 MIMO configuration
- 64 subcarriers
- 14 OFDM symbols per slot
- SNR range: 10-30 dB
- Static user scenario

```bash
python src/channel_generator.py
```

## Model Architecture
The beamforming model consists of:
- Input: Channel State Information (CSI)
- Hidden layers: CNN with attention mechanism
- Output: Beamforming weights

## XAI Implementation
Three main XAI approaches are used:
1. SHAP Values
2. Feature Attribution
3. Attention Visualization

## Performance Metrics
The system is evaluated using:
- SINR improvement
- Channel capacity
- Beamforming accuracy
- Decision interpretation clarity

## Usage

1. Generate Dataset:
```bash
python src/channel_generator.py
```

2. Analyze Channel:
```bash
python src/channel_analyzer.py
```

3. Train Model:
```bash
python src/training/trainer.py
```

4. Run XAI Analysis:
```bash
python src/xai/shap_analysis.py
```

## Results
- Expected SINR improvement: 15-20%
- Clear interpretability of beamforming decisions
- Fast adaptation to channel conditions

## Citation
```bibtex
@article{your_paper_2024,
  title={Explainable AI-Driven Beamforming for Adaptive MIMO Systems in 5G},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Sionna team for the channel simulation framework
- SHAP developers for XAI tools
