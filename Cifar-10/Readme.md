

This model has about %75 accuracy on Cifar-10 test set. Takes about 20 mins to train with GTX850M. Uses ~350M. Very small and fast model.
I used the sample codes from Stanford's CS231's Assignment 2. Thanks to them for releasing their broad knowledge.
http://cs231n.github.io/



Loss Function = Cross Entropy
Optimizer = SGD with Nesterov Momentum. Learning rate = 1e-3, L2 Regularization Scalar = 1e-4
Weight Initilization is implementation of He et al.'s work.	arXiv:1502.01852
Library used = PyTorch
