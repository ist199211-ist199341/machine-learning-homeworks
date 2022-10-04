# Machine Learning Homeworks

Homeworks of the Machine Learning course @ IST

## Setting Up Environment

Start by creating a Python virtualenv and installing dependencies:

```
python -m venv venv        # create the virtual env
source venv/bin/activate   # activate the virutal env
                           # (do this every time you open the project)
pip install -r requirements.txt
```

Finally, setup Git hooks in order for Notebook outputs to be cleared before commit:

```
pre-commit install
```
