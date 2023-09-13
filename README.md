# RHD Classification
Towards automatic diagnosis of rheumatic heart disease on echocardiographic exams through video-based deep learning

DOI: [10.1093/jamia/ocab061](https://doi.org/10.1093/jamia/ocab061)

# Project organization

```
├── README.md          <- The top-level README for developers using this project.
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── singularity        <- Singularity recipes to generate containers for reproducible code execution.
| 
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── dataprep       <- Scripts to cleanse or preprocess the downloaded data
│   │
│   ├── keras          <- Scripts to train models with Keras and useand then use to make predictions
│   │
│   └── utils          <- Utility scripts
|
└── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                          generated with `pip freeze > requirements.txt`
```

# Installing development requirements

    pip install -r requirements.txt

# Getting the pre-trained model weights

To download the weights for the Keras C3D model which was pre-trained on the Sports 1M dataset, simply execute the `src/utils/download_pretrained_model.sh` script. The original file name should remain unchanged and the immediate folder where the model can be found should be named `c3d`, which, in its turn, can be placed wherever you want, as long as you point to the correct path when executing the script.

This model was ported from one built with Caffe using the content available at [https://github.com/axon-research/c3d-keras](https://github.com/axon-research/c3d-keras).

# Containerization for easy setup and reproducible experiments

We are using Singularity to build container images and manage them, but you can use Docker and nvidia-docker if you prefer. The machine in which the container will be located only need the Nvidia driver and Singularity installed. Once the image is created, **no root access is needed**.

## Building the container image

We are using a pre-built environment created by the Tensorflow team and which can be found at their [Docker Hub repository](https://hub.docker.com/r/tensorflow/tensorflow/).

This Ubuntu 16.04 environment already comes bundled with everything needed to execute our neural network models, including `tensorflow-gpu 1.12.0` and `Python 3.5`.

To build the image simply execute the following command while in the repository root:

	sudo singularity build singularity/tf-1.12.0-gpu-py3.sqsh singularity/tf-1.12.0-gpu-py3.def 

 It may take a while as the image must be downloaded (only on the first time) and expanded, and the libraries specified in the definition file (`.def`) must be installed. The image created is in the SquashFS format and is read-only. In this format it is considerably smaller than in the ext3 format (`.img`), previously native to Singularity.

 For the execution of the Circulation models a `tensorflow 1.3.0` environment with `Python 2.7` must be used, and the image can be created with the respective definition file, found in the `singularity` folder.


We create the image with super user privileges and the `--writable` flag because this enables it to be incremented in the future.

## Running the container

To run the container as a shell in the current machine, simply execute:

	singularity shell --nv PATH/TO/IMAGE

The `--nv` flag is needed to allow communications with the Nvidia driver in the host machine, therefore enabling computations with your GPU.

# RHD classification code invocation example

The following is an invocation example for the main script from the repository's root: 

	python3 src/keras/classify_rhd.py --nn-type c3d --data-path data/preprocessed_rhd_videos --models-path models --results-path results/rhd --splits-file-path data/rhd_videos_viewpoint0_splits.csv --learning-rate 0.001 --num-epochs 50 --batch-size 16 --num-splits 10

To learn more about all the different parameters supported by the application, run the script with the `-h` or the `--help` flag.

# Singularity cheat sheet

## Creating container image

To create a writable container image for interactive development, use the following command:

	sudo singularity build --writable IMAGE-NAME.img DEFINITION-FILE.def

This will create an `ext3` container, which is can be updated. To create a compressed (many times smaller) and production-ready container, use the `.sif` (Singularity Image Format) file extension instead and drop the `--writable` flag.

## Enter container shell

	singularity shell --nv IMAGE-NAME.img

No root access is needed. The `--nv` flag is needed when NVIDIA GPU support inside the container.

## Update container contents

To update the content in a singularity container, open its shell as a super user with the `--writable` argument:

	sudo singularity shell --writable IMAGE-NAME.img

# Further questions

If the information you were looking for is missing from this README, please contact me at [joaofbsm@dcc.ufmg.br](joaofbsm@dcc.ufmg.br).