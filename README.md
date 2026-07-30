# Audio XAI

[[📄 poster](./poster%202026%20ICML%20v7.pdf)][[📜 link to article](https://arxiv.org/abs/2606.14466)][[ICML 2026](https://mlforaudioworkshop.github.io/accepted_submissions_2026/CameraReadys%204-83/13/CameraReady/paper_compressed.pdf)]

![PyPI version](https://img.shields.io/pypi/v/Audio-XAI.svg)
[![CodeQL](https://github.com/cncPomper/Audio-XAI/actions/workflows/codeql.yml/badge.svg)](https://github.com/cncPomper/Audio-XAI/actions/workflows/codeql.yml)
[![pages-build-deployment](https://github.com/cncPomper/Audio-XAI/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/cncPomper/Audio-XAI/actions/workflows/pages/pages-build-deployment)
[![Publish to PyPI](https://github.com/cncPomper/Audio-XAI/actions/workflows/publish.yml/badge.svg)](https://github.com/cncPomper/Audio-XAI/actions/workflows/publish.yml)
[![HuggingFace](https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/ARRSi/audio-xai)
[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/piotrkitowski/audio-xai-attack/data)

XAI for audio models

* [GitHub](https://github.com/cncPomper/Audio-XAI/) | [PyPI](https://pypi.org/project/Audio-XAI/) | [Documentation](https://cncPomper.github.io/Audio-XAI/)
* Created by [Piotr Kitłowski](https://audrey.feldroy.com/) | GitHub [@cncPomper](https://github.com/cncPomper) | PyPI [@pkitlo](https://pypi.org/user/pkitlo/)
* MIT License

## Data

https://www.kaggle.com/datasets/piotrkitowski/audio-xai-attack/data

## Features

### Explainers
* Grad-CAM (Gradient-weighted Class Activation Mapping)
* LRP (Layer-wise Relevance Propagation)

### Attackers
* PGD
* Xshift
* Psychoacoustic (ours)

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://cncPomper.github.io/Audio-XAI/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `master` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Setup

If you don't have `ffmpeg` installed, run:

### Windows 11

```
winget install -e --id Gyan.FFmpeg
```

### if you use conda, thats should enough
```
conda install -c conda-forge ffmpeg -y
```

Then verify the installation:
```
ffmpeg -version
ffprobe -version
```

## Running tensorboard

```
tensorboard --logdir Audio-XAI/runs/ --port 6006 --bind_all
```

## Examples

https://drive.google.com/drive/folders/1XD1rJwqSX-y-2OHzudYaatKE92MlYY8_?usp=sharing

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/Audio-XAI.git
cd Audio-XAI

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `audio_xai`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

Audio XAI was created in 2026 by Piotr Kitłowski.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.

## 1. General Information and Project Objective
The main objective of the project is to investigate the perceptual fragility of explanations (XAI methods) for deep learning models in the audio domain while keeping predictions unchanged.

## 2. Planned scope of experiments

- Datasets: Public datasets such as the Speech Commands Dataset (speech) and Sonics (synthetic/real music) will be used. The project will strictly ensure the immutability of the original data.
- Research models: Utilization and adaptation of audio recognition architectures: Audio Spectrogram Transformer, VGGish, Spectra, and ViT.
- XAI methods: Investigation of the vulnerability of gradient-based methods such as Grad-CAM and Integrated Gradients.
- Perceptual constraints: Instead of optimizing attacks against standard metrics, perceptual metrics will be considered (PESQ and STOI for speech, PEAQ for music).
- Computational resources and training: The project will require hardware acceleration (GPUs with a minimum of 16 GB VRAM). The estimated training and fine-tuning time for the base models is approximately 15 hours, while the main process of optimizing perceptual perturbations (XAI attack) for the entire test set is estimated to take an additional 25–30 hours of computation.

## 3. Planned Program Features

- **Classification and Attribution Module**: Reading models and generating explanation maps for them.
- **Perturbation module**: Generating subtle modifications to the audio signal with optimization that preserves high perceptual metrics (PESQ, PEAQ, STOI and ViSQOL).
- **Deployment and Automation**: Scripted building, testing, and deployment of applications using tools such as just and Python scripts built with typer or argparse.
- **Final deliverables**: The project will include clear documentation, user instructions, and tests relevant to the project’s scope.

## 4. Planned Technology Stack

The project will implement a robust base structure, automatically generated by tool `cookiecutter`.
- Environment management: Use of an isolated virtual environment managed by `uv` or `conda` (SLURM-managed cluster will be used).
- Code cleanliness: Enforced PEP8-compliant coding style with an increased line length limit. Syntax checking provided by an autoformatter and a linter using `ruff`.
- Version control: Rigorous use of a code repository with the [`conventional commits`](https://www.conventionalcommits.org/en/v1.0.0/) specification implemented.
- Frameworks and AI: Implementation of learning logic in dedicated frameworks such as `PyTorch Lightning` in conjunction with `Huggingface` libraries. Code used for experiments will be continuously exported from `Jupyter Lab` notebooks into structured library code.
- Experiments and configuration: Tracking progress, metrics, and logs using the `Tensorboard` platform. The configuration of model parameters and experiments will be completely separated from the execution code.
- Documentation: Use of `mkdocs` to fast and simple write documentation

## 5. Project schedule
| **Deadline dates in 2026** | **Planned scope of work and progress** | **Status** |
| :------------------------: | :--- | :---: |
| **30.03 - 05.04** | Repository configuration (Cookiecutter, Ruff, Uv). Defining the directory structure and ensuring that audio files remain immutable. | &#x2714; |
| **06.04 - 12.04** | Connecting W&B/TensorBoard. Training base classifiers using the PyTorch Lightning framework. (Estimated resource requirements: 15 hours of GPU computation) | &#x2714; |
| **13.04 - 19.04** | Implementation of explanation-generating (XAI) modules in clean code, after first exporting experiments from notebooks. Writing the first tests. | &#x2714; |
| **20.04 - 26.04** | Separating configuration from executable code. Preparing baseline attacks on attribution maps using standard distance metrics. | &#x2714; |
| **27.04 - 03.05** | Implementation of PESQ/STOI/PEAQ metric approximations directly into the attack optimization loop (generation of perceptual perturbations). | &#x2714; |
| **04.05 - 10.05** | Launch of the main research experiments on a dedicated cluster. (Estimated resource requirements: 25–30 hours of GPU computing for iterative processes). | &#x2714; |
| **11.05 - 17.05** | Scripting the execution of the entire experiment using the `just` tool and CLI libraries (e.g., `typer`). Aggregating tables containing the results. | &#x2714; |
| **18.05 - 21.05** | Finalization of the work: creating documentation and clear instructions for using the finished system. Organizing the code in accordance with PEP8. | &#x2714; |
| **21.05 - 25.05** | Preparation of the paper | &#x2714; |

---

<p align="left">
  <img src="PLGrid-logotype.png" alt="PLGrid logotype" width="150" height="150" align="left" hspace="20">
  <br><br><br>
  We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2026/019417
</p>
<br clear="left"/>

# Citation
If you find it useful, please consider citing our framework:


```
@misc{kitłowski2026perceivedfragilityexplanationsaudio,
      title={The Perceived Fragility of Explanations in Audio Models: Manipulation of Attribution with Unchanged Predictions}, 
      author={Piotr Kitłowski and Dominik Wiącek and Mateusz Modrzejewski},
      year={2026},
      eprint={2606.14466},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2606.14466}, 
}
```
