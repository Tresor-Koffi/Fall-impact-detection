# Machine Learning and Feature Ranking for Impact Fall Detection Event Using Multisensor Data

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20MMSP%202023-red.svg)](https://doi.org/10.1109/MMSP59012.2023.10337682)

**Authors:** Tresor Y. Koffi, Youssef Mourchid, Mohammed Hindawi, Yohan Dupuis  
**Affiliation:** CESI LINEACT Laboratory, UR 7527  
**Published:** IEEE 25th International Workshop on Multimedia Signal Processing (MMSP) 2023

---

## 📄 Abstract

Falls among individuals, especially the elderly population, can lead to serious injuries and complications. This work addresses the challenge of detecting **impact moments** within fall events using multisensor data. We apply thorough preprocessing techniques to the UP-FALL dataset, employ feature selection to identify relevant features, and evaluate various machine learning models for accurate impact detection. Our approach achieves **99.5% accuracy** with SVM, demonstrating the power of leveraging multisensor data for fall detection tasks.

## 🎯 Key Contributions

1. **Advanced Preprocessing Pipeline**: Novel preprocessing technique that eliminates noise and improves data quality by combining SMV (Signal Magnitude Vector) with threshold-based methods (2g threshold).

2. **Feature Selection Process**: Employed Random Forest-based feature ranking using Gini impurity to identify the top 5 most relevant features from 37 available features in the UP-FALL dataset.

3. **Machine Learning Comparison**: Evaluated 8 different ML algorithms for impact detection, with SVM achieving the highest accuracy (99.5%) and fastest inference time.

## 🚀 Key Results

| Algorithm | Accuracy | Recall | Precision | F1-Score | Training Time (s) |
|-----------|----------|--------|-----------|----------|-------------------|
| **SVM** | **99.50%** | 99.50% | 99.50% | 99.50% | 0.059 |
| Gradient Boosting | 99.35% | 98.65% | 98.84% | 98.74% | 0.844 |
| Random Forest | 99.28% | 98.47% | 99.18% | 98.47% | 0.607 |
| K-Nearest Neighbors | 98.35% | 98.45% | 99.03% | 98.74% | 0.005 |

## 📊 Dataset

This work uses the **UP-FALL Dataset** - a multimodal dataset for fall detection:
- **Activities**: 11 distinct fall activities + 6 Activities of Daily Living (ADLs)
- **Trials**: Each activity performed 3 times
- **Sensors**: Multiple accelerometers, gyroscopes, EEG sensors, and vision devices
- **Dataset Link**: [UP-FALL Dataset](https://sites.google.com/up.edu.mx/har-up/)

### Top 5 Selected Features (by Gini Impurity):
1. Ankle Accelerometer X-axis (Ank_ACCX)
2. Belt Accelerometer Y-axis (BeltACCY)
3. Ankle Accelerometer Y-axis (Ank_ACCY)
4. Neck Accelerometer Z-axis (NeckACCZ)
5. Belt Accelerometer Z-axis (BeltACCZ)

## 🛠️ Installation

### Prerequisites
- Python 3.6 or higher
- pip package manager

### Clone the Repository
```bash
git clone https://github.com/Tresor-Koffi/fall-detection-ml.git
cd fall-detection-ml
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

1. **Download the UP-FALL Dataset** from the official source and place it in the `data/` directory.

2. **Open the Jupyter Notebook**:
```bash
jupyter notebook impact_fall_detection.ipynb
3. **Run all cells** to reproduce the results from the paper.

### Pipeline Overview

The notebook implements the following pipeline:

```python
# 1. Data Loading
# Load UP-FALL multisensor data

# 2. Preprocessing
# - Z-score normalization
# - SMV (Signal Magnitude Vector) calculation
# - Threshold-based filtering (2g threshold)
# - Semi-automatic labeling with visual validation

# 3. Feature Selection
# - Random Forest feature importance
# - Select top 5 features based on Gini impurity

# 4. Model Training & Evaluation
# - Train 8 ML models (SVM, RF, GB, KNN, NB, DT, LR, SGD)
# - Evaluate using accuracy, precision, recall, F1-score
# - Generate ROC curves and confusion matrices
```

## 🔬 Methodology

### Preprocessing Pipeline
1. **Synchronization**: Align sensor data with visual data using timestamps
2. **Normalization**: Apply Z-score normalization to all 37 features
3. **SMV Calculation**: Compute Signal Magnitude Vector: `SMV = √(Ax² + Ay² + Az²)`
4. **Threshold Application**: Use 2g threshold to detect fall events
5. **Semi-automatic Labeling**: Validate impact detection with visual frames

### Feature Selection
- Method: Random Forest with Gini Impurity
- Selected: Top 5 features out of 37
- Reason: Balance between model complexity and performance

### Machine Learning Models
Evaluated 8 algorithms with optimized hyperparameters:
- Support Vector Machine (SVM)
- Random Forest (RF)
- Gradient Boosting (GB)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes (NB)
- Decision Tree (DT)
- Logistic Regression (LR)
- Stochastic Gradient Descent (SGD)

## 📈 Results

### Best Model: Support Vector Machine (SVM)
- **Accuracy**: 99.5%
- **AUC Score**: 1.000
- **Training Time**: 0.059s
- **Key Advantage**: Highest accuracy with fastest inference time

### Confusion Matrix Comparison
Our preprocessing method significantly reduces false positives and false negatives compared to standard preprocessing techniques.

## 📖 Citation

If you use this code or methodology in your research, please cite our paper:

```bibtex
@inproceedings{koffi2023machine,
  title={Machine Learning and Feature Ranking for Impact Fall Detection Event Using Multisensor Data},
  author={Koffi, Tresor Y. and Mourchid, Youssef and Hindawi, Mohammed and Dupuis, Yohan},
  booktitle={2023 IEEE 25th International Workshop on Multimedia Signal Processing (MMSP)},
  pages={1--6},
  year={2023},
  organization={IEEE},
  doi={10.1109/MMSP59012.2023.10337682}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or collaborations:
- **Tresor Y. Koffi**: ytkoffi@cesi.fr
- **CESI LINEACT Laboratory**: UR 7527

## 🔗 Links

- [Paper (IEEE Xplore)](https://doi.org/10.1109/MMSP59012.2023.10337682)
- [UP-FALL Dataset](https://sites.google.com/up.edu.mx/har-up/)
- [CESI LINEACT Laboratory](https://www.cesi.fr/)

---

**Note**: This repository contains the implementation of the research paper published at IEEE MMSP 2023. The code is provided for research and educational purposes.
