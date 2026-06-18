# computer-vision
Repository focused on implementation of papers written in the field of computer vision


---

## Python Version

Python 3.12.3

---

## Project Structure (Planned)

```bash
pytorch-fundamentals/
├── src/            # core implementations & importable code
├── notebooks/      # explanations & intuition
├── experiments/    # controlled experiments
└── requirements.txt
```

---

## Repository Structure Explained

### `src/`
Contains implementations of papers written in the field of computer vision. This is the core of the repository, focused on clean, minimal, and modular implementations without experimental or exploratory code. All code present inside src is importable.

### `notebooks/`
Used for exploration, intuition building, and analysis of concepts implemented in `src/`. Includes derivations, comparisons, and experiments to better understand behavior and design choices.

### `experiments/`
Houses structured experiments designed to evaluate the impact of different design choices.

### `requirements.txt`
Lists dependencies required to run the implementations, experiments, and notebooks.

**Note**:
- Often results of experiments will be stored in `results/` and `dataset/` will store datasets used as part of experiments and implementations. Both are part of `.gitignore`. 
- If you need any experiment results connect with me if I have them available on me I will provide them to you.

---

## Philosophy

- Implementation focused on explainability of how and why the models were implemented.
- Clear separation: code vs experiments vs validation  
- Minimal, explicit, and readable implementations
- Focus on understanding over optimization

---

## Make code importable

You need to make sure you run the following command in order to run listed notebooks, tests, experiments on your local setup.

```bash
pip install -e .
```

OR 

```bash
pip install --editable .
```

This allows to register `computer-vision` as an editable package with the help of `pyproject.toml` and allows the code present inside `src/` to be importable allowing for the notebooks, tests, experiments to run smoothly.

### Installation on Ubuntu/WSL

Recommended way:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Alternative (Not-Recommended)
If you are using Ubuntu (including WSL), system Python is externally managed and blocks pip install.

You can override this restriction:

```bash
pip install --break-system-packages -e .
```

Use this only if:
- You understand the implications
- You are working on a personal machine
- You are willing to repair or reset your Python environment if needed