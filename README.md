# NTIRE2024-AIG-VQA

## Test
* The pretrained model located in https://drive.google.com/drive/folders/1yvVsChygpX7aGkh43aZEMs3o8E4xwleJ?usp=sharing
* Download and put the pre-trained model in the root of project
* The test script implemented in the notebook file ``test.ipynb``
* Set the path of the file names and relevance prompts to test the test-dataset. 
```
with open('test.txt','r') as f:
    df_valid = f.readlines()
```
* Set the path of test-dataset in the CFG class in the video path :
```
class CFG:
    debug = False
    video_path = "<test dataset path>"
```