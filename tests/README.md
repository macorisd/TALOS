# Unit Tests for TALOS

This folder contains unit tests for the TALOS system. All tests are compatible with `pytest`.

> ⚠️ Some tests load the full models used in the TALOS pipeline, which may require significant memory. For this reason, I strongly recommend executing individual tests using:

```bash
pytest -k <test_name>
```

Before running the tests, make sure to install the testing dependencies:

```bash
pip install -r requirements-test.txt
```
