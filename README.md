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

Docs deploy automatically on push to `master` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

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

cluster. (Estimated resource requirements: 25–30 hours of GPU computing for iterative processes). |
| **11.05.2026 - 17.05.2026** | Scripting the execution of the entire experiment using the `just` tool and CLI libraries (e.g., `typer`). Aggregating tables containing the results. |
| **18.05.2026 - 24.05.2026** | Finalization of the work: creating documentation and clear instructions for using the finished system. Organizing the code in accordance with PEP8. Preparation of the paper(?) |

</div>
