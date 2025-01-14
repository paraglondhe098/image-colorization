# Image Colorization using GANs (Advanced computer vision project)
This project provides an implementation of Image colorization using GANs (Generative Adversarial Networks), specifically using a U-Net architecture 
for the generator and a PatchGAN for the discriminator. The project is constructed in Python with [PyTorch](https://pytorch.org/docs/stable/index.html) as the deep learning framework, and 
[PyTorch-Candle](https://github.com/paraglondhe098/pytorch-candle) for efficient training process.
## Project structure
Following are the main components:
- `notebook.ipynb`: This is the Jupyter notebook file containing the entire project implementation and execution steps.
- `README.md`: The file you are currently reading which contains detailed explanation about the project and the implementation.
- `requirements.txt`: It contains the list of required libraries and dependencies for the project.
- `callbacks.py`: It contains custom callback classes for saving images and checkpoints during the training process.
- `models.py`: It houses the neural network models used in this project.

### Training
During the training, the discriminator and the generator are trained alternately in the defined number of epochs. The progress of the training can be tracked and visualized.
### Callbacks
There are two callbacks employed:
1. `ImageSaver`: This callback is used to save the generated images after each epoch during training. As a result, we can review the progress of the generator in creating images.
2. `CheckpointSaver`: This callback is invoked to save the model after every specific number of steps. This can be particularly helpful during long training processes or when training is expected to be interrupted.

### Models
This project comprises of two main models, namely a generator and a discriminator model.
1. `Unet`: The generator model used is a U-Net model. This model is particularly effective for image-to-image translation problems.
2. `PatchDiscriminator`: The discriminator is implementing a PatchGAN style, which operates over patches of the image, rather than the whole image. It helps to make the model faster and more efficient.
3. `ImageColorizationModel`: The main model class holding the generator and discriminator models.

## Dataset
The data used for this project is available on [kaggle](https://www.kaggle.com/datasets/paraglondhe/coco-01-40k).

## Usage
In order to run the project, you need to have Python installed on your system. You can install the required libraries using the following command:
```bash
pip install -r requirements.txt
```
The code for the project is contained in the [`notebook.ipynb`](notebook.ipynb) Jupyter notebook file. You can run the notebook in a Jupyter notebook environment or on cloud platforms like Google Colab.
The implementation on Kaggle is available here: [Image Colorization on Kaggle](https://www.kaggle.com/code/paraglondhe/imagecol3)

## Future Objectives
The project's current configuration parameters, such as the number of epochs and batch size, are tailored for learning and experimentation.
These parameters may require further tuning based on the specific dataset and computational resources available. To achieve superior results, 
future work could include:

[//]: # (- Parameter Optimization: Experimenting with different configurations to find the optimal settings for better performance.)
- Advanced Techniques: Incorporating more sophisticated techniques to enhance the model’s accuracy and efficiency.
- Larger Datasets: Utilizing more extensive datasets to improve the generalization and robustness of the model.
- Extended Training Times: Allowing for longer training periods to enable the model to learn more complex patterns and features.

This project serves as a flexible foundation, open to modifications and enhancements. Users are encouraged to adapt and expand upon the provided code to explore the vast possibilities of conditional GANs and image colorization, aiming for innovative solutions and improved outcomes.