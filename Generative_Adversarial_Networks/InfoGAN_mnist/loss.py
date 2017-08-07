#Taken from CS231N's assignment notebooks.
def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751

    Inputs:
    - input: PyTorch Variable of shape (N, ) giving scores.
    - target: PyTorch Variable of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean BCE loss over the minibatch of input data.
    """
   
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()



def discriminator_loss(logits_real, logits_fake):
    
    
    #Computes the loss of discriminator according to Goodfellow et al. https://arxiv.org/abs/1406.2661

    
    loss_real = bce_loss(logits_real, Variable(torch.ones(logits_real.size())).type(dtype))
    loss_fake = bce_loss(logits_fake, Variable(torch.zeros(logits_fake.size())).type(dtype))
    loss = (loss_real + loss_fake)
    return loss

def generator_loss(logits_fake):
    
    #Computes the loss of generator according to Goodfellow et al. https://arxiv.org/abs/1406.2661
    
    
    loss = bce_loss(logits_fake, Variable(torch.ones(logits_fake.size())).type(dtype))
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.
    
    Input:
    - model: A PyTorch model that we want to optimize.
    
    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    return optimizer
