# Installation
Make a virtual environment:

```bash
python3 -m venv env
. env/bin/activate
```

Install PyTorch>=1.7 and torchvision>=0.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113
```

Install segment-anything:

```bash
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

Install additional tools:
```bash
pip install opencv-python pycocotools matplotlib tqdm
```