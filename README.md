# Attention-Pack

Official code for **"Attention-aware Inference Optimizations for Large Vision-Language Models with Memory-efficient Decoding"** (CVPR 2026).

📄 [Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Ilhan_Attention-aware_Inference_Optimizations_for_Large_Vision-Language_Models_with_Memory-efficient_Decoding_CVPR_2026_paper.pdf)

## Setup

```bash
conda create -n attpack python=3.10
conda activate attpack
cd src
bash setup.sh
```

For Video-LLaVA, follow the installation instructions in the [Video-LLaVA repository](https://github.com/PKU-YuanGroup/Video-LLaVA).

## Usage

All examples below use LLaVA-1.5-7B. `--rank_k` and `--rank_v` set the compression ranks for the key and value caches (R_k and R_v).

### OCR-VQA (R_k = 64, R_v = 64, with attention-aware decompression)

```bash
python inference/inference_ocrvqa.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --use-attpack \
    --rank_k 64 \
    --rank_v (64, 16) \
    --output-path ocrvqa_attpack.json
```

### A-OKVQA (R_k = 64, R_v = 64)

```bash
python inference/inference_aokvqa.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --use-attpack \
    --rank_k 64 \
    --rank_v 64 \
    --output-path aokvqa_attpack.json
```

### MMMU (R_k = 64)

```bash
python inference/mmmu/inference_mmmu.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --use-attpack \
    --rank_k 64 \
    --output-path mmmu_attpack.json
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ilhan2026attentionpack,
  title     = {Attention-aware Inference Optimizations for Large Vision-Language Models with Memory-efficient Decoding},
  author    = {Ilhan, Fatih and Liu, Gaowen and Kompella, Ramana Rao and Tekin, Selim Furkan and Huang, Tiansheng and Yahn, Zachary and Xu, Yichang and Liu, Ling},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
