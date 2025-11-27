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