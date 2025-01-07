# Code for "Decomposing and Interpreting Image Representations via Text in ViTs Beyond CLIP" (NeurIPS 2024)

This repository contains the code to reproduce our work "Decomposing and Interpreting Image Representations via Text in ViTs Beyond CLIP" which was accepted in NeurIPS 2024.

## Code Setup

```
conda create --name vit_decompose python=3.11
conda activate vit_decompose
pip install -r requirements.txt
```

## Dataset Setup

Specify the ImageNet/Waterbirds path in each script that you run.

You can download Imagenet-1000 from [here](https://huggingface.co/datasets/ILSVRC/imagenet-1k)

For the Waterbirds experiments, you can find our cleaned Waterbirds datasets in ``dataset_archives/`` 

CLIP aligner weights can be downloaded from [this link](https://drive.google.com/drive/folders/1LnkB6ncwRVeh5ZdckZ4qGQk8aP-yaguv?usp=sharing)

## Experiments

1. Component ablation : ``python component_ablation.py``

2. Image retrieval from image (wrt specified property) : ``python image_retrieval_from_image.py``

3. Image retrieval from text : ``python image_retrieval_from_text.py; python image_retrieval_from_text.py``

4. Image Segmentation: `` python zs_segmentation_decompose.py --model_key DINO --num_samples 5000``

5. Zero-shot spurous correlation mitigation: `` python zs_spur_correlation.py``

