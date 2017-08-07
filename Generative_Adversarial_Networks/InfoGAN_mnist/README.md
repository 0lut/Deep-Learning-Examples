This repo has not contained the complete code yet. I have got it working on my PC I have not uploaded the run.py yet.

This is implementation of famous GAN structure called InfoGAN in PyTorch framework.

Usage = (optirun) python run.py num_epoch=10 show_every=200 use_gpu=True

https://arxiv.org/pdf/1606.03657.pdf 

Layer sizes taken from C1.MNIST section. Paddings in generative module are quite confusing, since PyTorch does not include an implementation for "same" padding. If you want to train your model by using CPU's then you can change the padding=(1,1,0,0) in Conv2dTranspose layer. However, if you want to work with GPU's, cuDNN only supports padding with int. (I recall I did different paddings with GPU but I does not work atm.

I followed the IPython Notebook of Standford's CS231N class.* Thanks to them for broading their knowledge.











*I am not responsible of disciplinary actions that can be caused by using this code in your homework. You should take the Honor Code serious. I suffered from it a lot =)
