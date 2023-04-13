# VoMBaT: Visualisation of evaluation Measure Behaviour in Technology assisted reviews

## Installation

Create and activate conda environment:

```bash
$ conda create --name tar_metrics_demo python==3.10.6
$ conda activate tar_metrics_demo
```

Install Python requirements:

```bash
(tar_metrics_demo)$ pip install -r requirements.txt
```

No additional dependencies and data are required.
Datasets' parameters are defined in `data/datasets.json` file.

## Running

Start Streamlit server:

```bash
(tar_metrics_demo)$ streamlit run _🏠_Homepage.py
```

You can now access the app at http://localhost:8501
