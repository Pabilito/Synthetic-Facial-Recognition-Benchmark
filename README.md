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



