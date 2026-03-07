![Python](https://img.shields.io/badge/python-3.10+-blue)
[![University of Florence](https://i.imgur.com/1NmBfH0.png)](https://ingegneria.unifi.it)
Made by [Mattia Marilli](https://github.com/mattiamarilli) and [Marco Trambusti](https://github.com/MarcoTrambusti)

📄 **[Read the Full Report](https://github.com/MarcoTrambusti/ImageToLatex/blob/master/report/FML_Lab_Report.pdf)** 

# ImageToLatex

Project for **image → LaTeX formula** conversion built with PyTorch, using an **Encoder-Decoder** architecture:

- **CNN Encoder** (custom or ResNet18) to extract image features;
- **Transformer Decoder** to generate the LaTeX token sequence.

## Quick start (venv + requirements)

Clone the repository and move into the project directory:

```bash
git clone https://github.com/MarcoTrambusti/ImageToLatex.git
cd ImageToLatex
```

Then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## What the project does

The main script (`main.py`) runs the following steps:

1. Loads train/validation/test datasets from Hugging Face (`im2latex-100k`);
2. Applies image preprocessing (resize, tensor conversion, normalization);
3. Builds the vocabulary from training LaTeX tokens;
4. Trains the `Im2LatexModel` and tracks train/validation loss;
5. Saves the learning-curve plot to `learning_curve.png`;
6. Evaluates the model on the test set with official metrics.

## Notes

- The code automatically uses **GPU** if available, otherwise CPU; however, using a **GPU** is strongly recommended.
- Key parameters (batch size, epochs, image size, encoder selection) can be configured in `main.py`.

