# Synthetic Facial Recognition Benchmark

TBD


## Real Datasets Installation 

You can download LFW, CPLFW, CALFW and AgeDB-30 from kaggle:

~~~
pip install kaggle
kaggle datasets download -d yakhyokhuja/agedb-30-calfw-cplfw-lfw-aligned-112x112 -p ~/Downloads/
unzip ~/Downloads/agedb-30-calfw-cplfw-lfw-aligned-112x112.zip -d ~/data/face_datasets/
~~~

## Synthetic Datasets Installation 

Acquire synthetic datasets from their respective owners (sometimes it is a simple download, sometimes you need a permission) and extract them:

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


