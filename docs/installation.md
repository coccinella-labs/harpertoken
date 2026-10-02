# Installation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Setup

1. **Clone the project**:
    ```sh
    git clone https://github.com/coccinella-labs/harpertoken.git
    cd harpertoken
    ```

2. **Create Virtual Environment**:
    ```sh
    python3 -m venv venv
    ```

3. **Activate Virtual Environment**:
    ```sh
    source venv/bin/activate
    ```

4. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    # Optional: Install code quality tools (the ones pre-commit runs)
    pip install black flake8
    ```

## Requirements

- Python >= 3.10
- PyTorch with MPS support (Mac M1)
- 8GB RAM minimum
- Supported runtimes: macOS with Apple Silicon
