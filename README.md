# Addressee-Detection
## Dependencies
conda create -n Addressee_Detection python=3.7.9 anaconda  
conda activate Addressee_Detection  
pip install -r requirement.txt  

# install pytorch
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch

#Dataset Preparation
##generate single annotation .xlsx file each scenario from original Json MuMMER dataset 

1. python create_Annotation_file.py  --input inputpath --output MuMMER_Anno

here the input path is /media/fish/data2/MUMMER/MuMMER/annotations/ and the outputpath is ./data/MuMMER_Anno

2. ##Annotate the the .xlsx file based the annotation instruction.
based on the instruction for annotating the E-MUMMER is annotated and saved in ./data/E_MUMMER_anno_org


3. preprocess the annoation file and create single CSV file and split the dataset into training(60%, validation 20% and testng (20%)

python create_csv_preproces_dataset.py --input inputpath --output outputpath

here the inputpath is ./data/E_MUMMER_anno_org and the output patth is Single_Annotation_E_MUMMER_csv

4. crop face regions

python crop_face_region.py

5. trim audio segements

python trim_audio.py

#Training 
you can train ADNet in E-MuMMER end-to-end by using:
python exper2.py
#pretrained model
Our pretrained model performs 71% balanced accuracy in test set, you may check it running the following script

python Train_test.py --evalDataType test --savePath exps/pretrained/ --evaluation eval

