from pprint import pprint
import os
import setproctitle

class Config:
    name = 'COMP3314'
    model_type = 'unet'  # Options: 'deepcrack', 'unet', 'attention_unet', 'segformer', 'hnet'
    gpu_id = '0,1,2,3'

    setproctitle.setproctitle("%s" % name)

    #<---------- Paths and Directories ----------->#
    dir_img_tr = 'data/ISBI-2012-challenge/train-volume'
    dir_mask_tr = 'data/ISBI-2012-challenge/train-labels'
    dir_img_val = 'data/ISBI-2012-challenge/val-volume'
    dir_mask_val = 'data/ISBI-2012-challenge/val-labels'
    dir_img_test = 'data/ISBI-2012-challenge/test-volume'
    dir_mask_test = 'data/ISBI-2012-challenge/test-labels'
    dir_temp_tr = 'data/ISBI-2012-challenge/temp-tr'
    dir_temp_val = 'data/ISBI-2012-challenge/temp-val'
    dir_temp_ts = 'data/ISBI-2012-challenge/temp-ts'
    checkpoint_path = 'checkpoints/' # Checkpoint path for training
    log_path = 'log'
    pretrained_model = 'checkpoints/unet_fake.pth'  # Checkpoint path for testing

    #<-------------------------------------------->#
    
    #<--------------- Dataset Settings --------------->#
    keep_temp_tr = False  # Whether to keep temporary files for training
    keep_temp_val = False  # Whether to keep temporary files for validation
    
    
    #<---------- Preprocessing Settings ----------->#
    # Elastic deformation settings
    random_rotate = True      # whether to apply random rotation after elastic deformation
    use_elastic = True          # enable elastic deformations for training
    elastic_alpha = 40.0        # deformation intensity (pixels)
    elastic_sigma = 4.0         # Gaussian smoothing (pixels)
    elastic_n_copies = 100       # number of deformed copies per input image (approximate U-Net heavy augmentation)
    elastic_prob = 1.0          # probability to apply per copy
    #<-------------------------------------------->#
    
    
    #<---------- Training Settings ----------->#
    use_checkpoint = False
    # Treat 'epoch' as a safety cap; training stops early on plateau
    epoch = 100  # max epochs cap
    lr = 1e-2
    train_batch_size = 4
    val_batch_size = 4
    test_batch_size = 4
    
    val_every = 1 # legacy parameter for old train.py
    use_focal_loss = False  # legacy parameter for old train.py to use focal loss or not
    
    
    weight_decay = 5e-4
    lr_decay = 0.1
    momentum = 0.99
    
    # optimizer settings:
    sgd_params = {
        'lr': lr,
        'momentum': momentum,
        'weight_decay': weight_decay
    }
    #<-------------------------------------------->#

    #<------------- Loss Settings ---------------->#   
    ce_params = {
        # For multi-class CE: class weights [w_bg, w_fg, ...] or None. For binary, BCE pos_weight can be set.
        'weight': None,       # e.g., [0.5, 2.0]
        'ignore_index': -100,
        'reduction': 'mean',
        'pos_weight': None    # e.g., 2.0 for foreground upweight in binary BCE
    }
    #<-------------------------------------------->#
    
    
    # checkpointer
    save_format = ''
    save_acc = -1
    save_pos_acc = -1    # Model configuration
    
    # TensorBoard settings
    tensorboard_dir = 'runs/deepcrack'
    export_loss_dir = 'results/loss'  # Directory to save loss curve JPGs
    
    # Early stopping (until convergence / plateau)
    early_stopping = True
    es_patience = 3      # epochs without improvement
    es_min_delta = 0.0    # required improvement in val loss to reset patience

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def show(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')