# Saturation analysis

This repository contains my implementation on the following paper:

> Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, pages 249-256.


# Overview
The work discusses the incapability of certain activation functions to work on deep neural networks under certain initialization techniques. As the number of layers increase, a problem known as saturation arises.

Saturation is seen in bounded activation functions, as the weights in certain layers of network push all the activation values (i.e. the output of the activation functions) towards its bounded extremities. Since all the outputs in a certain layer are located in plateaus, learning hardly takes place.
<p align="center">
    <img width="400" src="https://artur-deluca.github.io/post/pretraining/figures/saturation.png"/>
</p>


# Experiments
The following results were obtained by training 5-hidden layer networks with 1000 neurons each, trained over the CIFAR-10 dataset. Here are some of the results:

<table style="width:100%">
  <tr>
    <th>Initialization</th>
    <th colspan="2">Activation function</th>
  </tr>
  <tr>
    <th></th>
    <td>Sigmoid</td>
    <td>Hyperbolic tangent</td>
  </tr>
  <tr>
    <td><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf">LeCun (1998)</a></td>
    <td><img src="./docs/imgs/sigmoid_lecun.gif"></td>
    <td><img src="./docs/imgs/tanh_lecun.gif"></td>
  </tr>
  <tr>
    <td><a href="http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf">Glorot and Bengio (2010)</a></td>
    <td><img src="./docs/imgs/sigmoid_glorot.gif"></td>
    <td><img src="./docs/imgs/tanh_glorot.gif"></td>
  </tr>
</table>

As we can observe, regardless of the method of initialization employed, the activation values of the fifth hidden layer in the sigmoid spike around zero. These results are in accordance with the ones shown in the paper.

The authors indicate that this abrupt shift can perhaps be explained by an attempt of the system to suppress the meaningless information fed by the previous layers, thus basically relying on its bias to make the classification. With this result, the authors show the unsuitability of the sigmoid in deep networks.

However, the results of the hyperbolic tangent function have shown to deviate from the original findings. Despite similar results in the baseline initialization (LeCun (1998)), these drastically differ in face of the initialization method proposed by the authors. The technique that allegedly inhibits saturation actually made it occur slightly faster.

These incoherences with the original findings are ought to be further analyzed. Moreover, despite the mentioned inadequacy of the sigmoid on deeper networks, there are ways to circumvent this problem. Early results of [Hinton et. al (2006)](https://www.mitpressjournals.org/doi/10.1162/neco.2006.18.7.1527) have shown that this problem can be avoided using unsupervised pretraining. 

More details on the historical perspective of saturation as well as a overview of unsupervised pretraining can be seen [here](https://artur-deluca.github.io/post/pretraining/)


# How to use it
1. Install the requirements
2. Run `train.py --help` to check the parameters
3. Have fun!

P.S.: there are more scripts that this, but these are supplementary. Feel free to use them as well.
