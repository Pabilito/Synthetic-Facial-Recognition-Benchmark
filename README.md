# Benchmarking Face Recognition without Real Faces - Replication Package

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)](https://arxiv.org/abs/TBD) 
[![DOI](https://img.shields.io/badge/DOI-TBD/TBD-blue)](https://doi.org/TBD)

Preprint available here: TBD

## Environment
Experiments were run on python 3.12 with the following dependencies:
~~~
pip install -r requirements.txt
~~~

## Real Datasets Installation 

You can download LFW, CPLFW, CALFW and AgeDB-30 from kaggle:

~~~
pip install kaggle
kaggle datasets download -d yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112 -p ~/Downloads/
unzip ~/Downloads/agedb-30-calfw-cplfw-lfw-aligned-112x112.zip -d ~/data/face_datasets/
~~~

## Synthetic Datasets Installation 

Acquire synthetic datasets from their respective owners (sometimes it is a simple download, sometimes you need a permission): 
- [SFace and SFace2](https://github.com/fdbtrs/SFace-Privacy-friendly-and-Accurate-Face-Recognition-using-Synthetic-Data?tab=readme-ov-file)
- [CemiFace](https://github.com/szlbiubiubiu/CemiFace)
- [DCFace](https://github.com/mk-minchul/dcface)
- [Vec2Face](https://github.com/HaiyuWu/Vec2Face)
- [IDiff-Face](https://github.com/fdbtrs/IDiff-Face)
- [ControlFace10k](https://huggingface.co/datasets/HuMInGameLab/ControlFace10K)
- [HyperFace](https://www.idiap.ch/paper/hyperface/)
- [Digi2Real](https://www.idiap.ch/paper/digi2real/)
- [LangevinDisco](https://www.idiap.ch/paper/synthetics-disco/)
- [SynFace](https://github.com/haibo-qiu/SynFace)
- [SynMulti-PIE](https://www.idiap.ch/en/scientific-research/data/synmultipie)
- MorphFace - contact authors

and extract them.
Some datasets (e.g., Vec2Face, Digi2Real) need additional preprocessing to match common structure of a folder per identity.


## Synthetic Datasets Preprocessing

Generate DATASETNAME_ann.txt files with mated and non-mated pairs for evaluation:
~~~
python data/synth/generate_ann.py
~~~

## Pretrained Models

Models trained on real data: 
- IResNet models are taken from [InsightFace](https://github.com/deepinsight/insightface/tree/master/model_zoo)
- EdgeFace and InceptionResnetV1 are loaded from libraries

Models trained on synthetic data:
- [SFace and SFace2](https://github.com/fdbtrs/SFace-Privacy-friendly-and-Accurate-Face-Recognition-using-Synthetic-Data?tab=readme-ov-file)
- [IDiff-Face](https://github.com/fdbtrs/IDiff-Face)
- Vec2Face, Langevin-Disco, Langevin-Dispersion and HyperFace are loaded from HuggingFace

## Simplified Execution
To re-run the experiments use: 

~~~
python run_all.py
~~~

Outputs are saved to the *Evaluation/* directory. It should be able to continue work where it finished it in case of a crash.

## Step by Step Execution

### Evaluation

To use real/synthetic datasets as benchmarks run: 
~~~
python eval_real.py
python eval_synth.py

# Since IJB works with templates, we report only biometric values and skip the accuracy
python eval_ijb.py
~~~

It saves separate comparisons results to: 
- *Evaluation/synth.csv* 
- *Evaluation/real.csv*  

and overal results to: 
- *Evaluation/synth_summary.csv* 
- *Evaluation/real_summary.csv*

To evaluate biometric metrics from saved similarity scores run:
~~~
# Checks Pearson coefficient
python eval_biometric.py

# Checks Spearman coefficient
python eval_rank_consistency.py  
~~~

Output:
- *Evaluation/biometric_metrics.csv*
- *Evaluation/rank_stability.csv*

### Correlation Analysis
~~~
# Aggregated evaluation
python eval_correlation.py
# Split by models trained only on real/synthetic data
python eval_correlation_by_model_type.py
# Confidence intervals
python eval_correlation_ci.py

# Biometric indicators evaluation
python eval_biometric_correlation.py
~~~

Outputs: 

- *Evaluation/correlation_all_datasets<_real|_synth>.png*
- *Evaluation/correlation_heatmap<_real|_synth>.png*
- *Evaluation/dataset_correlations<_real|_synth>.csv*

- *biometric_corr_<metric>.csv*         
- *biometric_corr_heatmap_<metric>.png*
- *biometric_corr_forest_<metric>.png*

## Supplementary Material

More detailed versions of the figures and tables presented in the paper are available [here](\Evaluation). This directory also stores scripts for data processing.
