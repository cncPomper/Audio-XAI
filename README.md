# Audio XAI

![PyPI version](https://img.shields.io/pypi/v/Audio-XAI.svg)

Python Boilerplate contains all the boilerplate you need to create a Python package.

* [GitHub](https://github.com/cncPomper/Audio-XAI/) | [PyPI](https://pypi.org/project/Audio-XAI/) | [Documentation](https://cncPomper.github.io/Audio-XAI/)
* Created by [Piotr Kitłowski](https://audrey.feldroy.com/) | GitHub [@cncPomper](https://github.com/cncPomper) | PyPI [@pkitlo](https://pypi.org/user/pkitlo/)
* MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://cncPomper.github.io/Audio-XAI/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

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
