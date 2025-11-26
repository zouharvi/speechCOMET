# speechCOMET

Then, this package can be used in Python with comet_early_exit package. The package name changed intentionally from Unbabel's package name such that they are not mutually exclusive.

## Development

Install the package locally and 
```bash
pip3 install -e .
speechcomet-train ...
speechcomet-score ...
```

or in Python:
```python
import speechcomet
model = speechcomet.download_model(speechcomet.load_from_checkcpoint("..."))
model.score(...)
```