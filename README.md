<div align="center">
<h1>MV-TAP: Tracking Any Point in Multi-View Videos</h1>

[**Jahyeok Koo**](https://scholar.google.com/citations?user=1Vl37dcAAAAJ&hl=ko)<sup>\*</sup> · [**Inès Hyeonsu Kim**](https://ines-hyeonsu-kim.github.io)<sup>\*</sup> · [**Mungyeom Kim**](https://github.com/mungyeom011) · [**Junghyun Park**](https://junghyun-james-park.github.io/)<br>[**Seohyun Park**](https://eainx.github.io/) · [**Jaeyeong Kim**](https://github.com/kjae0) · [**Jung Yi**](https://github.com/YJ-142150) · [**Seokju Cho**](https://seokju-cho.github.io) · [**Seungryong Kim**](https://cvlab.kaist.ac.kr/)

KAIST AI&emsp;&emsp;&emsp;

<span style="font-size: 1.5em;"><b>arXiv 2025</b></span>

<a href="https://github.com/cvlab-kaist/MV-TAP"><img src='https://img.shields.io/badge/arXiv-MVTAP-red' alt='Paper PDF'></a>
<a href='https://cvlab-kaist.github.io/MV-TAP/'><img src='https://img.shields.io/badge/Project_Page-MVTAP-green' alt='Project Page'></a>


</div>

**MV-TAP** (**T**racking **A**ny **P**oint in **M**ulti-view **V**ideos) is a robust point tracker that tracks points across multi-view videos of dynamic scenes by leveraging cross-view information. MV-TAP utilizes a cross-view attention mechanism to aggregate spatio-temporal information across views, enabling more complete and reliable trajectory estimation in multi-view videos

## TODO List
   - [x] Training & Evaluation Codes.
   - [ ] Model weights.
   - [ ] Kubric generation pipeline for Multi-view point tracking .
   - [ ] Harmony4D evaluation dataset and code.


## Environment

Prepare the environment by cloning the repository and installing the required dependencies:

```bash
conda create -y -n mvtap python=3.11
conda activate mvtap

# Use correct version of cuda for your system
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U lightning matplotlib mediapy einops wandb hydra-core imageio opencv-python omegaconf
```

## Evaluation

#### 0. Evaluation Dataset Preparation

For downloading Panoptic Studio and DexYCB datasets, please refer to official [MVTracker repository](https://github.com/ethz-vlg/mvtracker).

We will provide Kubric and Harmony4D evaluation datasts.

#### 1. Download Pre-trained Weights

To evaluate MV-TAP on the benchmarks, first download the pre-trained weights. [Link](https://drive.google.com/file/d/1Q-rqNl1ZkYhH4UtOjwcMH0oKkcCxMi7K/view?usp=sharing)

#### 2. Run Evaluation

To evaluate the Chrono, use the `experiment.py` script with the following command-line arguments:

```bash
python experiment.py mode=eval ckpt_path=/path/to/checkpoint
```


Replace `/path/to/checkpoint` with the actual path to your checkpoint file. This command will run the evaluation process and save the results in the specified `save_path`.

## Training

#### Training Dataset Preparation

Kubric train dataset generation pipeline will be provided



#### Run Training

```bash
python experiment.py
```

## 📚 Citing this Work
Please use the following bibtex to cite our work:
```
TBA
```

## 🙏 Acknowledgement
This project is largely based on the [CoTracker repository](https://github.com/facebookresearch/co-tracker) and [MVTracker repository](https://github.com/ethz-vlg/mvtracker). Thanks to the authors for their invaluable work and contributions.
