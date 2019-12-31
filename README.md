# Everybody Dance Faster 💃🏽🕺
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/release/python-370/)

[![Everybody Dance Faster flow](https://github.com/smellslikeml/motiontransfer/blob/master/assets/demo.gif)](https://www.youtube.com/watch?v=TXc6-ZTtlHw)

## Getting Started 
These instructions will show how to prepare your image data, train a model, and deploy the model to classify human action from image samples. See deployment for notes on how to deploy the project on a live stream.

### Prerequisites
## Software
- [Tensorflow 2.0](https://www.tensorflow.org) (not required on EdgeTPU)
- [EdgeTPU Python API](https://coral.ai/docs/edgetpu/api-intro) (for the EdgeTPU only)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [Google Cloud Storage Python client library](https://cloud.google.com/storage/docs/reference/libraries)

## Hardware
- [Coral Dev Board EdgeTPU](https://coral.ai/products/dev-board/)
- [GCP VM instance with GPU](https://cloud.google.com/compute/docs/gpus/)

### Install
We recommend using a virtual environment to avoid any conflicts with your system's global configuration. You can install the required dependencies via pip. The EdgeTPU will require the Python API as described above. 

This demo also requires that you have a Google Cloud account and have configured the python client with credentials. See ![this](https://cloud.google.com/compute/docs/tutorials/python-guide) for more resources.

## Data Acquisition
In the ```pose/``` directory, you will find all of the EdgeTPU resources required to capture training images from your booth participant. 

On the EdgeTPU, run:
```python3
python3 run.py
```
This script will generate the pose estimation overlays and raw image assets and send them to your declared Google Cloud Storage bucket.

## Training
After setting up a VM with a GPU instance in your Google Cloud account, install the requirements listed above. Then you can use the ```train.py``` script to train the Pix2Pix model.
```python3
python3 train.py
```

Periodic checkpoints will be stored in the ```checkpoints/``` directory. This script trains for 50 epochs to reduce training time.

## Generate Dance
We've included default source video frames from the [Bruno Mars - That's What I Like](https://www.youtube.com/watch?v=PMivT7MJ41M) music video. If you are using another source video, keep the framing, perspective, and background of your source in mind for better results.

To generate a dance gif of your participant, run the following on your Google Cloud training VM instance:
```python3
python3 generate_gif.py
```

The asset will be stored in the ```results/``` directory and in your declared Google Cloud Storage bucket.
## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details

## References

* [Blog Post](http://smellslikeml.com/everybody_dance_faster.html)
* [YouTube Video](https://www.youtube.com/watch?v=TXc6-ZTtlHw)
* [EdgeTPU PoseNet](https://github.com/google-coral/project-posenet)
* [Pix2Pix Colab Notebook](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)
* [Google Cloud Services](https://cloud.google.com)
