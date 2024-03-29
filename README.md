# Near-Duplicate Video Retrieval <br> with Deep Metric Learning
This repository contains the Tensorflow implementation of the paper 
[Near-Duplicate Video Retrieval with Deep Metric Learning](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w5/Kordopatis-Zilos_Near-Duplicate_Video_Retrieval_ICCV_2017_paper.pdf). 
It provides code for training and evalutation of a Deep Metric Learning (DML) network on the problem of Near-Duplicate 
Video Retrieval (NDVR). During training, the DML network is fed with video triplets, generated by a *triplet generator*.
The network is trained based on the *triplet loss function*.  The architecture of the network is displayed in the figure below.
For evaluation, *mean Average Precision* (*mAP*) and *Presicion-Recall curve* (*PR-curve*) are calculated.
Two publicly available dataset are supported, namely [VCDB](http://www.yugangjiang.info/research/VCDB/index.html) 
and [CC_WEB_VIDEO](http://vireo.cs.cityu.edu.hk/webvideo/).

<img src="https://raw.githubusercontent.com/MKLab-ITI/ndvr-dml/develop/train_net.png" width="70%">

## Prerequisites
* Python
* Tensorflow 1.xx

## Getting started

### Installation

* Clone this repo:
```bash
git clone https://github.com/MKLab-ITI/ndvr-dml
cd ndvr-dml
```
* You can install all the dependencies by
```bash
pip install -r requirements.txt
```
or
```bash
conda install --file requirements.txt
```

### Triplet generation

Run the triplet generation process for each dataset, VCDB and CC_WEB_VIDEO. This process will generate
two files for each dataset: 
1. the global feature vectors for each video in the dataset:     
\<output_dir>/\<dataset>_features.npy
2. the generated triplets:     
\<output_dir>/\<dataset>_triplets.npy

To execute the triplet generation process, do as follows:

* The code does not extract features from videos. Instead, the .npy files of the already extracted features have to be provided.
You may use the tool in [here](https://github.com/MKLab-ITI/intermediate-cnn-features) to do so.

* Create a file that contains the video id and the path of the feature file for each video in the processing dataset.
Each line of the file have to contain the video id (basename of the video file) 
and the full path to the corresponding .npy file of its features, separated by a tab character (\\t). Example:

        23254771545e5d278548ba02d25d32add952b2a4	features/23254771545e5d278548ba02d25d32add952b2a4.npy
        468410600142c136d707b4cbc3ff0703c112575d	features/468410600142c136d707b4cbc3ff0703c112575d.npy
        67f1feff7f624cf0b9ac2ebaf49f547a922b4971	features/67f1feff7f624cf0b9ac2ebaf49f547a922b4971.npy
                                                 ...	
		
* Run the triplet generator and provide the generated file from the previous step, 
the name of the processed dataset, and the output directory.
```bash
python triplet_generator.py --dataset vcdb --feature_files vcdb_feature_files.txt --output_dir output_data/
```

* The global video features extracted based on the [Intermediate CNN Features](https://github.com/MKLab-ITI/intermediate-cnn-features),
 and their generated triplets for both datasets can be found [here](https://drive.google.com/drive/folders/1swgdUsYwOy9pdoJzKfthAq74CqBEQDmA?usp=sharing).

### DML training

* Train the DML network by providing the global features and triplet of VCDB, 
and a directory to save the trained model.
```bash
python train_dml.py --train_set output_data/vcdb_features.npy --triplets output_data/vcdb_triplets.npy --model_path model/ 
```

* Triplets from the CC_WEB_VIDEO can be injected if the global features and triplet of the evaluation set
 are provide.
```bash
python train_dml.py --evaluation_set output_data/cc_web_video_features.npy --evaluation_triplets output_data/cc_web_video_triplets.npy --train_set output_data/vcdb_features.npy --triplets output_data/vcdb_triplets.npy --model_path model/
```

### Evaluation

* Evaluate the performance of the system by providing the trained model path and the global features of the 
CC_WEB_VIDEO.
```bash
python evaluation.py --fusion Early --evaluation_set output_data/cc_vgg_features.npy --model_path model/
````
OR
```bash
python evaluation.py --fusion Late --evaluation_features cc_web_video_feature_files.txt --evaluation_set output_data/cc_vgg_features.npy --model_path model/
```

* The *mAP* and *PR-curve* are returned

## Citation
If you use this code for your research, please cite our paper.
```bibtex
@inproceedings{kordopatis2017dml,
  title={Near-Duplicate Video Retrieval with Deep Metric Learning},
  author={Kordopatis-Zilos, Giorgos and Papadopoulos, Symeon and Patras, Ioannis and Kompatsiaris, Yiannis},
  booktitle={2017 IEEE International Conference on Computer Vision Workshop (ICCVW)},
  year={2017},
}
```
## Related Projects

**[Intermediate-CNN-Features](https://github.com/MKLab-ITI/intermediate-cnn-features)** - this repo was used to extract our features

**[ViSiL](https://github.com/MKLab-ITI/visil)** - video similarity learning for fine-grained similarity calculation

**[FIVR-200K](https://github.com/MKLab-ITI/FIVR-200K)** - download our FIVR-200K dataset

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details

## Contact for further details about the project

Giorgos Kordopatis-Zilos (georgekordopatis@iti.gr) <br>
Symeon Papadopoulos (papadop@iti.gr)
