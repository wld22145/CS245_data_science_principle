##Deep Learning Features

AWA2 deep learning features can be found in the link:

链接：https://pan.baidu.com/s/18x4Ze8bHNEY8mRqkfdX5tg 
提取码：2mos

Deep learning features are extracted by pre-trained neural network provided by PyTorch.

Dimensions of features are as follow:

ResNet152: 2048

Inception_v3: 2048

AlexNet: 9216 (256\*6\*6)

VGG19: 25088 (512\*7\*7)

VGG19_bn: 25088 (512\*7\*7)


##SIFT Features
AWA2 SIFT features can be found in the link:

链接：https://pan.baidu.com/s/1tO94vkazn9AAXBp6oHFL-A 
提取码：k7tj

SIFT features are extracted by algorithm in opencv-python.

sift224.py extracts SIFT key points and descriptors, then stores them in 
Pickle files.

sift224_demo.py gives an visualized example of SIFT.

sift.py and sift_demo.py can extract features from original images (not resized to (224,224)),
however, these features are too huge to deal with.

##Selective Search Features
AWA2 Selective Search features can be found in the link:



Selective Search features are extracted by selectivesearch. You can find more details in 
https://github.com/AlpacaDB/selectivesearch

selective_search.py extracts feature regions and corresponding labels, then 
stores them in Pickle files.

selective_search_demo.py gives an visualized example of Selective Search.





