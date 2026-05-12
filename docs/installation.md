# Installation

Move into cloned directory:
```sh
cd Audio-XAI
```

## Stable release

### uv
To install Audio XAI, run this command in your terminal:

```sh
uv add Audio-XAI
```

Or if you prefer to use `pip`:

```sh
pip install Audio-XAI
```

### conda
```sh
# conda may be heavy for some users, so it is OPTIONAL
conda create -n audio_xai python=3.12
conda activate audio_xai

# install package
pip3 install --upgrade pip
pip3 install -e . 
```

## From source

The source files for Audio XAI can be downloaded from the [Github repo](https://github.com/cncPomper/Audio-XAI).

You can either clone the public repository:

```sh
git clone https://github.com/cncPomper/Audio-XAI
```

Or download the [tarball](https://github.com/cncPomper/Audio-XAI/tarball/master):

```sh
curl -OJL https://github.com/cncPomper/Audio-XAI/tarball/master
```

Once you have a copy of the source, you can install it with:

```sh
cd Audio-XAI
uv sync
```
