# MOBA MO

## System Dependencies

All PufferLib Ocean environments require the GNU Scientific Library (GSL):

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

After installing GSL, rebuild environments:
```bash
pip install -e .
```
