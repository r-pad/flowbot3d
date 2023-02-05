# Contributing

## Install
```
pip install -e ".[develop]"

pre-commit install
```

## Run CI locally
To run the CI locally:

Setup (make sure docker is installed):
```
brew install act
```

Run act
```
act -j develop
```
