<div align="center">

# INSID3: Training-Free In-Context Segmentation with DINOv3

<p align="center">
  <a href="LINK_TO_PAPER"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=flat-square&labelColor=444444" alt="Paper arXiv"></a>
  <a href=https://visinf.github.io/INSID3/><img src="https://img.shields.io/badge/Project-Page-1f6feb?style=flat-square&labelColor=444444" alt="Project Page"></a>
</p>

**[Claudia Cuttano](https://scholar.google.com/citations?user=W7lNKNsAAAAJ)<sup>1,2</sup> ·
[Gabriele Trivigno](https://scholar.google.com/citations?user=JXf_iToAAAAJ)<sup>1</sup> ·
[Christoph Reich](https://scholar.google.com/citations?user=0OYstzAAAAAJ&hl=de)<sup>2,3,5,6</sup> ·
[Daniel Cremers](https://scholar.google.com/citations?user=cXQciMEAAAAJ&hl=en)<sup>3,5,6</sup> ·
[Carlo Masone](https://scholar.google.com/citations?user=cM3Iz_4AAAAJ)<sup>1</sup> ·
[Stefan Roth](https://scholar.google.com/citations?user=0yDoR0AAAAAJ&hl=en)<sup>2,4,5</sup>**

<sup>1</sup> Politecnico di Torino &nbsp;&nbsp; 
<sup>2</sup> TU Darmstadt &nbsp;&nbsp; 
<sup>3</sup> TU Munich &nbsp;&nbsp; 
<sup>4</sup> hessian.AI &nbsp;&nbsp; 
<sup>5</sup> ELIZA &nbsp;&nbsp; 
<sup>6</sup> MCML  

✨ **CVPR 2026** ✨

</div>

INSID3 solves in-context segmentation entirely within a single frozen DINOv3 backbone:

🚀 **Training-free:** no fine-tuning, no segmentation decoder, no auxiliary models   
🔍 **New insight:** identifies and removes a positional bias in DINOv3   
📈 **State-of-the-art, smaller & faster:** outperforms both training-free and specialized methods while using a single backbone  
🌍 **Generalizes broadly:** from object-level to part-level and personalized segmentation, across natural, medical, underwater and aerial domains  

<p align="center">
  <img src="assets/teaser.png">
</p>

## ⚙️ Environment Setup

To get started, create a Conda environment and install the required dependencies.  
INSID3 is compatible with **PyTorch ≥ 2.0**. The experiments in the paper were run with **PyTorch 2.7.1 (CUDA 12.6)**, which we provide as a reference configuration.

To set up the environment using Conda, run:

```bash
conda create --name insid3 python=3.10 -y
conda activate insid3
pip install -r requirements.txt
```

## 🧱 DINOv3 Weights

INSID3 relies on a **frozen DINOv3 backbone**. Please download the pretrained weights from the official repository: 👉 https://github.com/facebookresearch/dinov3

Create the ```pretrain``` directory:

```bash
mkdir -p pretrain
```

Place the weights of the backbone you want to use in the ```pretrain/``` folder:

```
pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```
By default, we use the Large model (```dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth```). 


## 📍 Minimal Usage

Here is a minimal example to segment a target image given a reference image and its mask:

```python
from models import build_insid3

# Build model
model = build_insid3()

# Set reference and target
model.set_reference("path/to/ref_image.png", "path/to/ref_mask.png")
model.set_target("path/to/tgt_image.png")

# Predict
pred_mask = model.segment()  # (1024, 1024) bool
```


## 📦 Data

Please refer to [docs/data.md](docs/data.md) for dataset preparation instructions.

## 🚀 Inference

Evaluate INSID3:

```bash
python inference.py --dataset coco --exp-name insid3-coco
```

#### Main arguments:

- `--dataset`: supported [`coco`, `lvis`, `pascal_part`, `paco_part`, `isaid`, `isic`, `lung`, `suim`, `permis`]
- `--model_size`: DINOv3 backbone size (`small`, `base`, `large`, default: `large`)

- `--shots`: number of reference images per episode (e.g., 1-shot, 5-shot, default: 1)


- Other args: hyperparameters (e.g., `--tau`, `--merge-thresh`, `--svd-comps`) have default values as in the paper; pass them to override the defaults. See `opts.py`.



**Note:** By default, the predicted mask is upsampled to the original image resolution using **bilinear interpolation**. For additional refinement, enable **CRF-based refinement** with `--crf-mask-refinement`.
 
 To install the CRF refinement dependency, clone and install the CRF package:



```bash
git clone https://github.com/netw0rkf10w/CRF.git
cd CRF
python setup.py install
cd ..
```

## 💡 Why INSID3 Works

INSID3 builds on two key observations about DINOv3 features.

(i) **Dense DINOv3 features** naturally induce a **structured decomposition of the scene**. By clustering them, we obtain coherent object- and part-level regions without supervision.

<p align="center">
  <img src="assets/clustering.png" alt="Dense DINOv3 features" width="80%">
</p>


(ii) Besides semantic matches, DINOv3 also **responds to absolute image position**. Given a patch on the bird’s tail in the reference image, the DINOv3 similarity map activates on (i) the tail in the target image, but also (ii) over the left portion of the image.
<p align="center">
  <img src="assets/similarity_maps.png" alt="Dense DINOv3 features" width="55%">
</p>

PCA on low-semantic-content images reveals that this effect lives in a stable **low-dimensional subspace**. INSID3 removes it in a training-free way: we identify the positional component of DINOv3 features and project onto its **orthogonal complement**. This suppresses coordinate-driven responses while preserving semantics.

<p align="center">
  <img src="assets/positional_subspace.png" alt="Dense DINOv3 features" width="80%">
</p>

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{cuttano2026insid3,
  title     = {{INSID3}: Training-Free In-Context Segmentation with {DINOv3}},
  author    = {Claudia Cuttano and Gabriele Trivigno and Christoph Reich and Daniel Cremers and Carlo Masone and Stefan Roth},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgements

We gratefully acknowledge the contributions of the following open-source projects:

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [Matcher](https://github.com/aim-uofa/Matcher)
