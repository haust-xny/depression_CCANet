### Introduction

The Application of CCANet Network - Diagnosis of Depression

Using Datasets：**AVEC2014**
Dataset Download Address:https://pan.baidu.com/s/1ZZPuHLYBntngzRpnjQWmtg
Note that the extraction password can be found in Reply #4 of the journal email.


Pretreatment process：

​	1.**sampling**，Sampling AVEC2013, taking 100 frames from each video, preserving the original label

​	2.**Face alignment and cropping**，Using the **MTCNN** tool

### File Introduction

```
preprocess.py	Mainly used for preprocessing video information, extracting frames from it, and extracting faces from video frames
generate_label_file():	Merge the shipped labels into a CSV file
get_img():	Extract video frames, with each video extracting 100-105 frames at intervals
get_face():	Using MTCNN to extract faces and segment images

CCANet.py	The network structure of the model
```

```
load_data.py	Obtain the image storage path and corresponding labels to it
writer.py	Create Tensorboard recorder to save training process losses
dataset.py	Inheriting torch.utils.Dataset, responsible for converting data into iterators that torch.utils.data.DataLoader can handle
train.py	model training
validate.py	Verification Model
test.py		Test the performance of the model and record the prediction score, which is saved in testInfo.csv. Record the path, label, and prediction score of each image
main.py		Model training entry file
```

```
img		Extracted video frame files
log		Tensorboard log file
model_dict	Log file trained model parameter file
processed	Store facial images and label files after preprocessing
AVEC2014	Dataset storage location
```

```
To view training logs:
After installing the Tensorboard library, enter the command Tensorboard --lofdir log_Dir_Path, open the URL that appears after the command is executed
Log_ Dir_ Path is the folder path where Tensorboard log files are stored
```

```
Run Test Sequence:preprocess.py--->test.py
Our model training results are available for download in Releases, the file name is: CCANet_result.pth, after downloading it, put it in the model_dict folder, you can run test.py
```
