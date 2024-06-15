# Digit-recognition
## General info
That project was created to learn more about PyTorch and image classfication. This was a simple project, but I got a overall knowlege about creating project like that
in Pytorch. Dataset, which I used is well-known MNIST dataset. The goal of the project was to detect the hand-written numbers from 0 to 9.

## Model
I created a simple model with two convolutional layers and one pooling layer. Below is presented architecture of the model:
<pre>
ModelMnist1(
    (block1): Sequential(
        (0): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1))
        (1): ReLU()
        (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (classifier): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=1440, out_features=10, bias=True)
    )
)
</pre>
## Results
I ommitted the hyperparemeter tunning phase, because I got a really good results. I checked four metrics: accuracy, precision, recall, f1-score. The results on test set:
* Precision: 98.15
* Recall: 98.15
* F1score: 98.15

For further analys feel free to check the notebook.

## Techonologies
The project was created with:
* PyTorch
* Matplotlib
* Pandas
* Seaborn

## Status
The project has been completed.
