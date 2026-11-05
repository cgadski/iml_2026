# IML 2026 Assignments

This repository has the homework assignments for _Introduction to Machine Learning_ from the winter semester of 2026.

## Quickstart

Install [uv](https://github.com/astral-sh/uv) and run `uv sync` to install dependencies into a virtual environment.

I highly recommend working with notebooks. One simple option is to use JupyterLab in your web browser. (Launch with `uv run jupyter lab`.) Many editors also support working with Jupyter notebooks. You generally first have to install a kernel spec. (For example: `uv run -m ipykernel install --user --name iml`.)

Assignments are `.pdf` problem sheets in this directory, like `homework_1.pdf`. For your convenience, I'm also generating Python notebooks you can use for your submission. These notebooks are available in [percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html). To turn them into Jupyter notebooks (`.ipynb`) you can run the `%.ipynb` Makefile target. You can export either format as a html document with the `build/%.html` target.

## Submission Instructions

To submit a homework assignment, just fill out a notebook and export to html. Make sure to write your names. If you modified any helper code in `iml/`, please also submit this. Otherwise, just submitting an exported notebook is enough.
