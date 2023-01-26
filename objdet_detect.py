# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import torch
from easydict import EasyDict as edict

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# model-related
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.models.fpn_resnet import get_pose_net
from tools.objdet_models.resnet.models.fpn_resnet import BasicBlock
from tools.objdet_models.resnet.models.fpn_resnet import Bottleneck



from tools.objdet_models.resnet.models.fpn_resnet import PoseResNet as pose_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing 
from tools.objdet_models.resnet.utils.torch_utils import _sigmoid

from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2



# load model-related parameters into an edict
def load_configs_model(model_name='darknet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()  

    # get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))    
    
    # set parameters according to model type
    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False
        configs.min_iou = 0.5

    elif model_name == 'fpn_resnet':
        ####### ID_S3_EX1-3 START #######     
        #######
        print("student task ID_S3_EX1-3")
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')
        configs.arch = 'fpn_resnet'
        
        # configs.pin_memory = True
        # configs.distributed = False  # For testing on 1 GPU only

        configs.input_size = (608, 608)
        configs.hm_size = (152, 152)
        configs.down_ratio = 4
        configs.max_objects = 50
        configs.conf_thresh = 0.5
        configs.imagenet_pretrained = False
        configs.head_conv = 64
        configs.num_classes = 3
        configs.num_center_offset = 2
        configs.num_z = 1
        configs.num_dim = 3
        configs.num_direction = 2  # sin, cos
        configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
        }
        configs.num_input_features = 4

        #######
        ####### ID_S3_EX1-3 END #######     

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU
    configs.no_cuda = True # if true, cuda is not used
    configs.gpu_idx = 0  # GPU index to use.
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    return configs


# load all object-detection parameters into an edict
def load_configs(model_name='fpn_resnet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()    

    # birds-eye view (bev) parameters
    configs.lim_x = [0, 50] # detection range in m
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0] # reflected lidar intensity
    configs.bev_width = 608  # pixel resolution of bev image
    configs.bev_height = 608 

    # add model-dependent parameters
    configs = load_configs_model(model_name, configs)

    # visualization parameters
    configs.output_width = 608 # width of result image (height may vary)
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]] # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

    return configs


# create model according to selected model type
def create_model(configs):

    # check for availability of model file
    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

    # create model depending on architecture name
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
    
    elif 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid')
        
        ####### ID_S3_EX1-4 START #######     
        #######
        print("student task ID_S3_EX1-4")
        #############################################
        
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}
        num_layers = 18
        
        model = get_pose_net(num_layers, configs.heads, configs.head_conv, configs.imagenet_pretrained)                 #
        #############################################
        #######
        ####### ID_S3_EX1-4 END #######     
    
    else:
        assert False, 'Undefined model backbone'

    # load model weights
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_filename))

    # set model to evaluation state
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)  # load model to either cpu or gpu
    model.eval()          

    return model


# detect trained objects in birds-eye view
def detect_objects(input_bev_maps, model, configs):

    # deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():  

        # perform inference
        outputs = model(input_bev_maps)

        # decode model output into target object format
        if 'darknet' in configs.arch:

            # perform post-processing
            output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])    

        elif 'fpn_resnet' in configs.arch:
            # decode output and perform post-processing
            
            ####### ID_S3_EX1-5 START #######     
            #######
            print("student task ID_S3_EX1-5")

            # print(f'out:{outputs}')
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # print(f'out_sigmoid : {outputs}')
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'], outputs['dim'], K=40)
            # print(f'asdf : {detections}')
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs)            
            detections = detections[0][1]
            # print(f'alalal : {ret}')
            #######
            ####### ID_S3_EX1-5 END #######     

            
    # print(f'~~~~~~~~~~~~~: {detections}')
    ####### ID_S3_EX2 START #######     
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = [] 
    # print(f'detections.shape : {detections.shape}')
    # print(f'ret.shape : {len(ret)}')
    
    ## step 1 : check whether there are any detections
    if detections != []:
        ## step 2 : loop over all detections
        for row in detections:
            _id, _x, _y, _z, h, _w, _l, _yaw = row
            
            ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
            x=_y/configs.bev_height*(configs.lim_x[1]-configs.lim_x[0]) + configs.lim_x[0]
            y=_x/configs.bev_width*(configs.lim_y[1]-configs.lim_y[0]) + configs.lim_y[0]
            w=_w/configs.bev_width*(configs.lim_y[1]-configs.lim_y[0])
            l=_l/configs.bev_height*(configs.lim_x[1]-configs.lim_x[0])
            
            
            
            # x = (_x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
            # y = (_y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
            # # z = _z - configs.lim_z[0]
            # w = _w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
            # l = _l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
            _yaw = - _yaw


            ## step 4 : append the current object to the 'objects' array
            # corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
            objects.append([1, x, y, _z, h, w, l, _yaw])
    #######
    ####### ID_S3_EX2 START #######   
    
    return objects    



























# # sol

# # ---------------------------------------------------------------------
# # Project "Track 3D-Objects Over Time"
# # Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
# #
# # Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
# #
# # You should have received a copy of the Udacity license together with this program.
# #
# # https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# # ----------------------------------------------------------------------
# #

# # general package imports
# import numpy as np
# import torch
# from easydict import EasyDict as edict

# # add project directory to python path to enable relative imports
# import os
# import sys
# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# # model-related
# from tools.objdet_models.resnet.models.fpn_resnet import get_pose_net
# from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing 

# from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
# from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2
# from tools.objdet_models.resnet.utils.torch_utils import _sigmoid

# # load model-related parameters into an edict
# def load_configs_model(model_name, configs=None):

#     # init config file, if none has been passed
#     if configs==None:
#         configs = edict()  

#     # get parent directory of this file to enable relative paths
#     curr_path = os.path.dirname(os.path.realpath(__file__))
#     parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))    
    
#     # set parameters according to model type
#     if model_name == "darknet":
#         configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
#         configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
#         configs.arch = 'darknet'
#         configs.batch_size = 4
#         configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
#         configs.conf_thresh = 0.5
#         configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
#         configs.pin_memory = True
#         configs.distributed = False  # For testing on 1 GPU only
        
#     elif  model_name == 'fpn_resnet':
#         ####### ID_S3_EX1-3 START #######     
#         #######
#         print("student task ID_S3_EX2")
#         #######
#         ####### ID_S3_EX1-3 END ####### 
#         configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet')
#         configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')
#         configs.arch='fpn_resnet'
#         configs.input_size = (608, 608)
#         configs.hm_size = (152, 152)
#         configs.down_ratio = 4
#         configs.max_objects = 50
#         configs.conf_thresh = 0.5
#         configs.num_layers=18
#         configs.imagenet_pretrained = False
#         configs.head_conv = 64
#         configs.num_classes = 3
#         configs.num_center_offset = 2
#         configs.num_z = 1
#         configs.num_dim = 3
#         configs.num_direction = 2  # sin, cos

#         configs.heads = {
#             'hm_cen': configs.num_classes,
#             'cen_offset': configs.num_center_offset,
#             'direction': configs.num_direction,
#             'z_coor': configs.num_z,
#             'dim': configs.num_dim
#         }
#         configs.num_input_features = 4

#         #######
#         ####### ID_S3_EX1-3 END #######     

#     else:
#         raise ValueError("Error: Invalid model name")
    
#     # GPU vs. CPU
#     configs.no_cuda = True # if true, cuda is not used
#     configs.gpu_idx = 0  # GPU index to use.
#     configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

#     return configs

# def load_configs(model_name='fpn_resnet', configs=None):

#     if configs==None:
#         configs = edict()    

#     # birds-eye view (bev) parameters
#     configs.lim_x = [0, 50] # detection range in m
#     configs.lim_y = [-25, 25]
#     configs.lim_z = [-1, 3]
#     configs.lim_r = [0, 1.0] # reflected lidar intensity
#     configs.bev_width = 608  # pixel resolution of bev image
#     configs.bev_height = 608 

#     # add model-dependent parameters
#     configs = load_configs_model(model_name, configs)

#     # visualization parameters
#     configs.output_width = 608 # width of result image (height may vary)
#     configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]] # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

#     return configs

# # create model according to selected model type
# def create_model(configs):
#     assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

#     # create model depending on architecture name
#     if (configs.arch == 'darknet') and (configs.cfgfile is not None):
#         print('using darknet')
#         model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
    
#     elif 'fpn_resnet' in configs.arch:
#         print('using ResNet architecture with feature pyramid')
#         ####### ID_S3_EX1-4 START #######     
#         #######
#         print("student task ID_S3_EX1-4")
#         model=get_pose_net(configs.num_layers,configs.heads,configs.head_conv,configs.imagenet_pretrained)
#         #######
#         ####### ID_S3_EX1-4 END #######     
    
#     else:
#         assert False, 'Undefined model backbone'

#     # load model weights
#     model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
#     print('Loaded weights from {}\n'.format(configs.pretrained_filename))

#     # set model to evaluation state
#     configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
#     model = model.to(device=configs.device)  # load model to either cpu or gpu
#     model.eval()          

#     return model


# # detect trained objects in birds-eye view
# def detect_objects(input_bev_maps, model, configs):

#     # deactivate autograd engine during test to reduce memory usage and speed up computations
#     with torch.no_grad():  

#         # perform inference
#         outputs = model(input_bev_maps)

#         # decode model output into target object format
#         if 'darknet' in configs.arch:

#             # perform post-processing
#             output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
#             detections = []
#             for sample_i in range(len(output_post)):
#                 if output_post[sample_i] is None:
#                     continue
#                 detection = output_post[sample_i]
#                 for obj in detection:
#                     x, y, w, l, im, re, _, _, _ = obj
#                     yaw = np.arctan2(im, re)
#                     detections.append([1, x, y, 0.0, 1.50, w, l, yaw])    

#         elif 'fpn_resnet' in configs.arch:
#             # decode output and perform post-processing
            
#             ####### ID_S3_EX1-5 START #######     
#             print("student task ID_S3_EX1-5")
#             print(outputs)
#             outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
#             outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
#             detections=decode(outputs['hm_cen'],outputs['cen_offset'],outputs['direction'],outputs['z_coor'],outputs['dim'],K=40)
#             #print(detections.shape)
#             #print(outputs)
#             #print(detections)
#             detections=detections.cpu().numpy().astype(np.float32)
#             #print(detections)
#             detections=post_processing(detections,configs)
#             #print(detections)
#             detections=detections[0][1]
#             #print(detections)


#             #######
#             ####### ID_S3_EX1-5 END #######  
    
#     ####### ID_S3_EX2 START #######     
#     #######
#     # Extract 3d bounding boxes from model response
#     print("student task ID_S3_EX2")
#     objects = [] 

#     # ## step 1 : check whether there are any detections
#     if detections !=[]:
#         ## step 2 : loop over all detections
#         for obj in detections:
#             ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
#             id,_x,_y,_z,h,_w,_l,_yaw=obj
#             x=_x/configs.bev_height*(configs.lim_x[1]-configs.lim_x[0])
#             y=_y/configs.bev_width*(configs.lim_y[1]-configs.lim_y[0])-(configs.lim_y[1]-configs.lim_y[0])/2
#             w=_w/configs.bev_width*(configs.lim_y[1]-configs.lim_y[0])
#             l=_l/configs.bev_height*(configs.lim_x[1]-configs.lim_x[0])
#             # x = (_y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
#             # y = (_x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
#             # z = _z - configs.lim_z[0]
#             # w = _w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
#             # l = _l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
#             _yaw = -_yaw

#             ## step 4 : append the current object to the 'objects' array
#             # if x>= configs.lim_x[0] and x<=configs.lim_x[1] and y>=configs.lim_y[0] and y<=configs.lim_y[1]:
#             objects.append([1,x,y,_z,h,w,l,_yaw])
#     #######
#     ####### ID_S3_EX2 END #######   
    
#     return objects   
