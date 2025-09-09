# Data science tasks

## Project folder Structure

```text
├── docs
│   ├── task_1_pipeline.jpg
│   └── task_2_pipeline.jpg
├── notebooks
│   └── cross_reference_builder.ipynb
├── pipeline
│   ├── __init__.py
│   ├── task_1_pipeline.py
│   └── task_2_pipeline.py
├── resources
│   ├── export
│   │   └── .gitkeep
│   ├── task_1
│   │   ├── .gitkeep
│   │   ├── supplier_data1.xlsx
│   │   └── supplier_data2.xlsx
│   ├── task_2
│   │   ├── .gitkeep
│   │   ├── reference_properties.tsv
│   │   └── rfq.csv
├── utilities
│   ├── checks.py
│   ├── data_utils.py
│   ├── file_handler.py
│   ├── __init__.py
│   ├── log.py
│   └── similarity.py
├── .gitignore
├── README.md
├── requirements_simple.txt
├── requirements.txt
├── setup.cfg
├── task_1.py
└── task_2.py
```

## Setup


Clone the repository using the following command:

```shell
git clone github.com:tarzahedi/data-science-assessment.git
cd data-science-assessment
```

For this project, I'm using Python version 3.12. It's recommended to use pyenv
to switch between different python versions using the following command:

```shell
# If python 3.12 does not exist:
pyenv install 3.12

# Enable python 3.12
pyenv shell 3.12

# Make sure you're using the correct version:
python3 --version
```

Create a virtual environment in the root folder of the project:

```shell
python3 -m venv venv
```

Activate the virtual environment:

```shell
# Using Mac or Linux
source ./venv/bin/activate

# For windows
venv\Scripts\activate
```

Use `requirements.txt` file to install dependencies:

```shell
pip install -r requirements.txt
```

**Note**: I used `pip freeze > requirements.txt` to generate the requirements file.
This will generate requirements based on all packages that are installed in the active
environment. You can also use `requirements_simple.txt` file which is less strict about
exact versioning if you encountered any problem during the installation.

## Data

The input data sources are not part of the project. Please make sure
that you download and put them as resources folder:

```text
repository_root
├── resources
│   ├── export                          # Exports go here
│   │   └── inventory_dataset.csv
│   │   └── top3.csv
│   ├── task_1                          # Task_1 files
│   │   ├── supplier_data1.xlsx
│   │   └── supplier_data2.xlsx
│   └── task_2                          # Task_2 files
│       ├── reference_properties.tsv
│       └── rfq.csv

```

## Usage

For each task I created a notebook and also a pipeline based on Prefect that can run
independently.

Make sure that you put the input data in resources folder.

### Pipelines (Prefect)

To run the pipeline using Prefect. You can start the Prefect UI:

```shell
# Make sure prefect is using local version
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

prefect server start
```

The local prefect server will be available at `http://localhost:4200`.

Prefect is useful to scheduling and monitoring of the pipelines and you can also use
cloud-based services to deploy your pipelines.

To run task pipeline you should run the following commands in the root folder:

```shell
# Make sure that prefect is running and API URL is set.
# prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

# Running the pipeline for task_1
python task_1.py

# Running the pipeline for task_2
python task_2.py
```

### Notebooks

You can also run the solutions using the notebook files. The notebooks exist in
`notebooks` folder.

After activating and installing the requirements run the following commands:

```shell
cd ./notebooks
jupyter-notebooks .
```

You can now open the notebook that you want and run it inside jupyter notebook.

### Streamlit

To run the streamlit app:

```bash
streamlit run app.py
```
