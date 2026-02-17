# PufferLib Ocean Environments

## System Dependencies

Multiobjective environments require the **GNU Scientific Library (GSL)**.

### Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install libgsl-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install gsl-devel
```

**macOS:**
```bash
brew install gsl
```

### Building

After installing GSL, build the environments:

```bash
pip install -e .
```

## Additional Dependencies

Individual environments may have additional system dependencies. Check environment-specific READMEs for details:
