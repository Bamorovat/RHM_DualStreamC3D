"""
The Config file for the RHM dataset

The Dataset list is: RHM
The View Status list is NormalFrame, MotionAggregation, FrameVariationMapper, DifferentialMotionTrajectory,
Normal, Subtract, OpticalFlow, MotionHistoryImages
The View List is: FrontView, BackView, OmniView, RobotView
The Model_name List is: DualStreamC3D, SlowFast_Multiview
The Number of classes is: 14

Author: Mohammad Hossein Bamorovat Abadi
Email: m.bamorovvat@gmail.com

License: GNU General Public License (GPL) v3.0
"""

params = dict()

params['dataset'] = 'RHM'   # Dataset name
params['view1'] = 'OmniView'   # View1 name
params['view2'] = 'OmniView'   # View2 name
params['view1_status'] = 'Normal'   # View1 status
params['view2_status'] = 'DifferentialMotionTrajectory'  # View2 status
params['model_name'] = 'DualStreamC3D'  # Model name
params['num_classes'] = 14  # Number of classes
params['epoch_num'] = 1  # Number of epochs
params['batch_size'] = 1   # Batch size
params['step'] = 10    # Step size for learning rate decay in optimizer
params['num_workers'] = 1  # Number of workers for dataloader
params['learning_rate'] = 1e-4  # Learning rate
params['momentum'] = 0.9    # Momentum for optimizer
params['weight_decay'] = 4e-5   # Weight decay for optimizer
params['display'] = 50  # Display interval
params['pretrained'] = False    # Use pretrained model
params['gpu'] = [0]     # GPU ID
params['clip_len'] = 16     # Clip length
params['frame_sample_rate'] = 1     # Frame sample rate
params['useTest'] = True  # See evolution of the test set when training
params['nTestInterval'] = 10  # Run on test set every nTestInterval epochs


