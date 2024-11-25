# Cat Dog Classifier

A refined classifier designed to distinguish between cats and dogs, built with PyTorch and fine-tuned on the ResNet-50 architecture. Optimized for inference using Apple's Metal Performance Shaders (MPS), it efficiently leverages macOS hardware acceleration for superior performance. For environments without Apple Silicon, the model seamlessly supports inference on CUDA-enabled devices, ensuring versatility across platforms.

---

## Features

- **High Accuracy**: This model scored 98% on training accuracy and 98% on test accuracy.
- **Hardware Acceleration**: Utilizes Apple's MPS for fast inference on macOS devices.
- **PyTorch Ecosystem**: Built using PyTorch, torchvision, and torchaudio.


---

## Training

- **Platform**: Trained using Google Colab on A100 with 40GB VRAM
- **Epochs**: Trained on 2 classes with convergence at 10 epochs.
- **Dataset**: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

---

## Requirements

Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.

### Dependencies

The project uses the following main libraries:
- Python 3.9
- PyTorch 2.5.1
- torchvision 0.20.1
- torchaudio 2.5.1
- OpenCV 4.10.0
- NumPy 1.26.4
- Pillow 11.0.0

For a full list of dependencies, refer to the `environment.yml` file.

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ooopssieee/catdogclassifier.git
   cd catdogclassifier

2. **Create the conda environment**
    conda env create -f environment.yml
    conda activate catdog

3. **For non-apple silicon machines**
    In 'inference.py', change your device type to check for 'cuda' rather than mps.

---

## Limitations

- **Other class**: Predicts 'Other' when the confidence of cat or dog is lower than the confidence threshold. The model was not trained on 3 classes, this is only a binary classification. Henceforth, 'Other' prediction might not always be as robust as the prediction of other two classes.
