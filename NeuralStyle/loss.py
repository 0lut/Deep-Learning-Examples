"""
Loss functions that used in Neural style.
Implemented in PyTorch
"""
def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).
    
    Returns:
    - scalar content loss
    """
    loss = torch.sum(torch.pow((content_current - content_original),2)) * content_weight
    return loss
    
    def gram_matrix(features, normalize=True):
      """
      Compute the Gram matrix from features.

      Inputs:
      - features: PyTorch Variable of shape (N, C, H, W) giving features for
        a batch of N images.
      - normalize: optional, whether to normalize the Gram matrix
          If True, divide the Gram matrix by the number of neurons (H * W * C)

      Returns:
      - gram: PyTorch Variable of shape (N, C, C) giving the
        (optionally normalized) Gram matrices for the N input images.

      """
      N,C,H,W = features.size()
      features = features.view(N*C, H*W)
      gram = torch.mm(features, features.t())
      if normalize:
          gram /= (H*W*C)

      return gram
      
def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Variable giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Variable holding a scalar giving the style loss.
    """
    feat_gram = list(map(gram_matrix,feats))
    loss = Variable(torch.zeros(1))
    for i in range(len(style_weights)):
        loss += (style_weights[i] * torch.sum((torch.pow(style_targets[i] - feat_gram[style_layers[i]], 2))))
        
    return loss
    
    
    def tv_loss(img, tv_weight):
      """
      Compute total variation loss.

      Inputs:
      - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
      - tv_weight: Scalar giving the weight w_t to use for the TV loss.

      Returns:
      - loss: PyTorch Variable holding a scalar giving the total variation loss
        for img weighted by tv_weight.
      """
      # Your implementation should be vectorized and not require any loops!
      N,C,H,W = img.size()
      tvloss = Variable(torch.zeros(1))
      tvloss = torch.sum(torch.pow(img[:, :, 0:H-1,:] - img[:, :, 1:H,:], 2))
      tvloss += torch.sum(torch.pow(img[:, :, :, 0:W-1] - img[:, :, :, 1:W], 2))
      tvloss *= tv_weight
      return tvloss

    
    
    
