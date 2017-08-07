def build_classifier():
    
    return nn.Sequential(
        ###########################
        ######### TO DO ###########
        #args should be replaced  #
        ###########################
        ###########################
        Unflatten(batch_size, 1, 28, 28),
        nn.Conv2d(1, 32, kernel_size=5, stride=1),
        nn.LeakyReLU(1e-2),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, kernel_size=5, stride=1),
        nn.LeakyReLU(1e-2),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(1024, 1024),
        nn.LeakyReLU(1e-2),
        nn.Linear(1024, 1),
    )

def build_dc_generator(noise_dim=NOISE_DIM):
        return nn.Sequential(nn.Linear(noise_dim,1024),
                         nn.ReLU(inplace=True),
                         nn.BatchNorm1d(1024),
                         nn.Linear(1024, 7*7*128),
                         nn.BatchNorm1d(7*7*128),
                         Unflatten(-1, 128, 7, 7),
                         nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                         nn.BatchNorm2d(64),
                         nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
                         nn.Tanh(),
                         Flatten()
                                        
    )
