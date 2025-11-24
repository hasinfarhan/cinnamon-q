# Setting Up Python 3.11 for pipenv

## Why Python 3.11?

Python 3.12 has compatibility issues with `alpaca-trade-api` due to urllib3 dependency conflicts. Python 3.11 is the recommended version for this project.

## Installation Options

### Option 1: System-wide Installation (Ubuntu/Debian)

**For Ubuntu 24.04 (Noble) and newer:**
Python 3.11 is not in the default repositories. Use the deadsnakes PPA:

```bash
# Add deadsnakes PPA
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils

# Verify installation
python3.11 --version
```

**For Ubuntu 22.04 (Jammy) and older:**
Python 3.11 may be available in default repos or universe:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

### Option 2: Using pyenv (Recommended)

pyenv allows you to manage multiple Python versions easily:

```bash
# Install pyenv (if not already installed)
curl https://pyenv.run | bash

# Add to your shell profile (~/.bashrc or ~/.zshrc)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Install Python 3.11
pyenv install 3.11.9

# Set as local version for this project
cd /root/projects/cinnamon-q
pyenv local 3.11.9
```

## Setting Up pipenv with Python 3.11

After installing Python 3.11:

### If virtualenv doesn't exist yet:
```bash
cd /root/projects/cinnamon-q
pipenv install --python 3.11
```

### If virtualenv already exists (with Python 3.12):
```bash
cd /root/projects/cinnamon-q
# Remove the old virtualenv
pipenv --rm

# Create new virtualenv with Python 3.11
pipenv install --python 3.11
```

### Verify the Python version:
```bash
pipenv run python --version
# Should output: Python 3.11.x
```

## Pipfile Configuration

The `Pipfile` has been updated to specify Python 3.11:
```toml
[requires]
python_version = "3.11"
```

This ensures that anyone running `pipenv install` will use Python 3.11.

## Troubleshooting

### "python3.11: command not found"
- Make sure Python 3.11 is installed (see installation options above)
- Verify with: `python3.11 --version`

### "pipenv: command not found"
- Install pipenv: `pip install pipenv` or `pip3 install pipenv`

### Virtualenv still using wrong Python version
- Remove and recreate: `pipenv --rm && pipenv install --python 3.11`
- Check which Python pipenv is using: `pipenv --py`

## Next Steps

Once Python 3.11 is set up:
1. Install dependencies: `pipenv install`
2. Verify imports work: `pipenv run python -c "import alpaca_trade_api; print('OK')"`
3. Run the bot: `pipenv run python main.py`

