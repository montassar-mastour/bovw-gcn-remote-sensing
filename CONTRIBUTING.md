# Contributing Guidelines

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Test thoroughly
6. Submit a pull request

## Development Setup
```bash
git clone https://github.com/YOUR_USERNAME/bovw-gcn-remote-sensing.git
cd bovw-gcn-remote-sensing
pip install -r requirements.txt
pip install -e .
pip install pytest black flake8
```

## Code Style

- Follow PEP 8
- Use Black formatter: `black .`
- Check with flake8: `flake8 .`
- Type hints encouraged

## Commit Messages

Use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Formatting
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance

Examples:
```
feat: add multi-head attention layer
fix: resolve CUDA memory leak in GCN
docs: update training guide with GPU requirements
```

## Pull Request Process

1. Update documentation
2. Add/update tests
3. Ensure tests pass
4. Update CHANGELOG.md
5. Request review

## Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=.

# Run specific test
pytest tests/test_models.py::test_gcn_layer
```

## Questions?

Open an issue for discussion!