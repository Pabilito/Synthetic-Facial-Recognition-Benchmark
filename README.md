# Synthetic Facial Recognition Benchmark

Reproduction package for TBD

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

and extract them:

~~~
# You can preview structure of the files with preview.sh script. 
# If it matches structure_preview.txt, you can use directly extract .zip and .tar.gz files.

chmod +x preview.sh
./preview.sh 
              
chmod +x extractor.sh
./extractor.sh  

# In case of line endings compatibility problems use:
dos2unix filename.sh
# or
sed -i 's/\r//' ./filename.sh
~~~

Some datasets (Vec2Face, Digi2Real) need additional preprocessing to match common structure of a folder per identity
~~~
TBD
~~~

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

## Evaluation

To use real/synthetic datasets as benchmarks run: 
~~~
python eval_real.py
python eval_synth.py
~~~

It saves separate comparisons results to: 
- *Evaluation/synth.csv* 
- *Evaluation/real.csv*  

and overal results to: 
- *Evaluation/synth_summary.csv* 
- *Evaluation/real_summary.csv*

## Demographic Analysis

We use predictors from [DeepFace](https://github.com/serengil/deepface) to estimate age and gender of each dataset identity.

~~~
# All datasets
python Dataset_Demographic_Analysis/demographic_analysis.py
# Selected dataset
python Dataset_Demographic_Analysis/demographic_analysis.py --datasets LFW CemiFace
~~~

## Correlation Analysis
~~~
# Aggregated evaluation
python eval_correlation.py
# Split by models trained only on real/synthetic data
python eval_correlation_by_model_type.py
# Confidence intervals
python -u eval_correlation_ci.py
~~~

Outputs: 

- *Evaluation/correlation_all_datasets[_real|_synth].png*
- *Evaluation/correlation_heatmap[_real|_synth].png*
- *Evaluation/dataset_correlations[_real|_synth].csv*

TODO:
Find correct links for hyperface and langevins