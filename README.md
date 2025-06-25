<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/DanielSFlamarich/stock-seasonality">
    <img src="data/external/img/some_logo_or_image.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Stock Seasonality Analysis</h3>

  <p align="center">
    In this README you'll find everything you need to install this project
    <br />
    <a href="https://github.com/DanielSFlamarich/stock-seasonality"><strong>Explore the docs »</strong></a>
    <br />
    <a href="https://github.com/DanielSFlamarich/stock-seasonality/issues">Report Bug</a>
    ·
    <a href="https://github.com/DanielSFlamarich/stock-seasonality/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- GETTING STARTED -->
## About the Project
**Stock Seasonality Analysis** is a research-focused Python project designed to explore **seasonal patterns in equities and ETFs** using publicly available data from [Yahoo Finance](https://finance.yahoo.com/).

There’s a large body of evidence showing the **limits of predictability in financial markets**, including:

-   The **Efficient Market Hypothesis (EMH)**

-   [_The Adaptive Markets Hypothesis_ (Lo, 2004)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4283395/)

-   Studies demonstrating that most professional fund managers fail to consistently outperform benchmarks.

The aim is to:
-   Load, clean, and visualize historical data for a variety of tickers

-   Spot **recurring seasonal trends** — such as monthly, quarterly, or annual behaviors

-   Support **curiosity-driven exploration** across a broad universe of stocks, including unfamiliar tickers
- Understanding **structure and historical behavior**, not forecasting the future

-  Building **testable, modular functions** and a simple codebase with robust tooling

<!-- GETTING STARTED -->
## Getting Started
This section helps you get the project up and running locally, step by step.

### Prerequisites

To set up your environment reliably, we recommend the following:

-   **Python 3.11+** (managed via [`pyenv`](https://github.com/pyenv/pyenv))

-   [`uv`](https://github.com/astral-sh/uv) (fast Python package manager. Please do not use Conda; everytime anyone uses it a kitty dies)

-   [`pre-commit`](https://pre-commit.com/) (for automated linting and cleaning)

-   [`JupyterLab`](https://jupyterlab.readthedocs.io/en/stable/) (already included via dependencies, recommended for dashboard visualizations in notebooks)

##### Install dependencies:
On macOS:
`brew install pyenv uv pre-commit`

`uv` installation for Linux:
`curl -LsSf https://astral.sh/uv/install.sh | sh`

### Installation
##### 1. Clone the repo
`git clone https://github.com/DanielSFlamarich/stock-seasonality.git`
`cd stock-seasonality`

##### 2. **Install Python version** (if not already):
`pyenv install 3.11.9`
`pyenv local 3.11.9`

##### 3. Set up virtual environment
`uv venv`
`source .venv/bin/activate`
`uv pip sync requirements.lock`
>This will install all the pinned dependencies exactly as specified, ensuring reproducibility across environments.


##### 4. Set up pre-commit hooks
`pre-commit install`
>This registers the hooks defined in .pre-commit-config.yaml to run automatically before every commit (e.g. Black, Ruff, isort, and internal ones)



##### 5.  (Optional) Update dependencies
_If you've changed the list of top-level dependencies in `requirements.in`, regenerate the lock file:_
`uv pip compile requirements.in`
>This keeps your environment deterministic and reproducible.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

How to use this repo (WIP).
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are great and an amazing way to learn and make the project better. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please clone the repo and create a pull request.

1. Clone the repo.
2. Create your Feature Branch (`git checkout -b feature/CreatingAmazingFeature`)
3. Commit your Changes (`git commit -m 'Added some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/CreatingAmazingFeature`)
5. Open a Pull Request

It's that simple! :smile:

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License


<p align="right">(<a href="#readme-top">back to top</a>)</p>
