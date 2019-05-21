'''
run.py

@yashbonde - 14.05.2019
'''

# keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
from math import ceil
import argparse
import cv2
import pafy
import logging

# custom
from model import SSD7
from layers import L2Norm, AnchorBoxes
from ssd_utils import BatchGenerator, SSDLoss
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--url', type = str, help = 'train URL')
parser.add_argument('--train', type = bool, default = False, help = 'bool whether training or not')
parser.add_argument('--labels', type = str, help = 'path to labels file')
parser.add_argument('--batch-size', type = int, default = 16, help = 'mini batch size')
parser.add_argument('--epochs', type = int, default = 10, help = 'number of training epochs')
parser.add_argument('--model', type = str, default = None, help = 'path to model')
parser.add_argument('--opdir', type = str, default = './tests', help = 'folder for writing logs')
parser.add_argument('--debug', type = bool, default = True, help = 'give debug outputs')
args = parser.parse_args()
print(args)

if not args.train and (args.model is None and args.url is None):
    # testing mode requires the following things:
    # 1. model
    # 2. URL of video to stream
    raise ValueError('Testing mode requires Model path and URL')

if args.train and not args.labels:
    # training mode requires following items:
    # 1. Path to labels.csv
    raise ValueError('Need to provide labels when training')

if args.debug:
    logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s - %(message)s')
    logging.debug('Setting Logging Mode to DEBUG')

# config
logging.debug('Starting Model Building')
img_height = 360 # height of the input images
img_width = 480 # width of the input images
img_channels = 3 # number of color channels
n_classes = 2 # number of positive classes
batch_size = args.batch_size # training batch size
epochs = args.epochs # number of epochs to train the model (only for training)

# clear previous models from memory
logging.debug('Building the model (this may take some time)...')
K.clear_session()
model = SSD7(min_scale = 0.1,
             max_scale = 0.9,
             scales = [0.08, 0.16, 0.32, 0.64, 0.96],
             aspect_ratios_global = [0.5, 1.0, 2.0],
             aspect_ratios_per_layer = None,
             two_boxes_for_ar1 = True,
             limit_boxes = True,
             variances = [1.0, 1.0, 1.0, 1.0],
             coords = 'centroids',
             normalize_coords = False)
model.build_model(image_size = (img_height, img_width, img_channels),
                  num_classes = n_classes)
logging.debug('... Complete!')

# ========== MODEL TRAINING ========== #
if args.train:
    # encoder to make ground truth into SSD loss format
    logging.debug('Starting Box Encoder')
    predictor_sizes = model.predictor_sizes
    ssd_box_encoder = SSDBoxEncoder(img_height = img_height,
                                    img_width = img_width,
                                    n_classes = n_classes, 
                                    predictor_sizes = predictor_sizes,
                                    scales = [0.08, 0.16, 0.32, 0.64, 0.96],
                                    aspect_ratios_global = [0.5, 1.0, 2.0],
                                    aspect_ratios_per_layer = None,
                                    two_boxes_for_ar1 = True,
                                    steps = None,
                                    offsets = None,
                                    limit_boxes = False,
                                    variances = [1.0, 1.0, 1.0, 1.0],
                                    pos_iou_threshold = 0.5,
                                    neg_iou_threshold = 0.2,
                                    coords = 'centroids',
                                    normalize_coords = False)

    # generator
    logging.debug('Making BatchGenerator and defining generator function')
    train_dataset = BatchGenerator(labels_path = args.labels,
                                   box_output_format = ['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])
    train_generator = train_dataset.generate(batch_size = args.batch_size,
                                             shuffle = True,
                                             train = True,
                                             ssd_box_encoder = ssd_box_encoder,
                                             equalize = False,
                                             brightness = (0.5, 2, 0.5),
                                             flip = 0.5,
                                             translate = ((5, 50), (3, 30), 0.5),
                                             scale = (0.75, 1.3, 0.5),
                                             include_thresh = 0.4)
    logging.debug('')
    # define call backs if in training mode and train
    callbacks = [ModelCheckpoint('./model/ssd7_model_epoch{epoch:02d}_loss{loss:.4f}.h5',
                                 monitor = 'val_loss',
                                 verbose = 1,
                                 save_best_only = True,
                                 save_weights_only = True,
                                 mode = 'auto',
                                 period = 1),
                 EarlyStopping(monitor = 'val_loss',
                               min_delta = 0.001,
                               patience = 2),
                 ReduceLROnPlateau(monitor = 'val_loss',
                                   factor = 0.5,
                                   patience = 0,
                                   epsilon = 0.001,
                                   cooldown = 0)]

    model.train_generator(train_generator, epochs = epochs, callbacks = callbacks)

    # save the model
    model_name = 'ssd7'
    model.model.save('{}.h5'.format(model_name))
    model.model.save_weights('{}_weights.h5'.format(model_name))

    logging.debug('Model saved under {}.h5'.format(model_name))
    logging.debug('Weights also saved separately under {}_weights.h5'.format(model_name))

# ========== MODEL DEPLOYMENT ========== #
else:
    logging.debug('Loading weights from {}'.format(args.model))
    model.model.load_weights(args.model)

    # basic checkups
    vPafy = pafy.new(args.url) 
    play = vPafy.getbest(preftype = "webm") # make streaming object
    logging.debug('URL: {}'.format(args.url))

    # make video object
    logging.debug('Streaming the Video')
    video = cv2.VideoCapture(play.url)
    fps = video.get(cv2.CAP_PROP_FPS)
    logging.warning('FPS of video: {}'.format(fps))

    # stream the video and dump the image
    frame_num = 0
    classes = ['Pink Panther', 'Little Man']
    colors = {'Pink Panther': (0,0,0), 'Little Man': (0,0,255)}
    min_count, sec_count = 0, 0
    while True:
        # get frame and increment number
        ret, frame = video.read()
        if not ret: break
        frame_num += 1

        if int(frame_num % fps) == 0:
            # do predcitions and save
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            frame -= 127.5
            frame /= 127.5
            frame = np.expand_dims(frame, axis = 0)
            y_pred = model.model.predict([frame])
            y_pred_decoded = decode_y2(y_pred,
                                       confidence_thresh = 0.5,
                                       iou_threshold = 0.4,
                                       top_k = 'all',
                                       input_coords = 'centroids',
                                       normalize_coords = False,
                                       img_height = None,
                                       img_width = None)

            # make the frame in old format
            frame = np.squeeze(frame)
            frame *= 127.5
            frame += 127.5
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.uint8)

            # add to image
            for box in y_pred_decoded[0]:
                pt_min = (int(box[-4]), int(box[-3]))
                pt_max = (int(box[-2]), int(box[-1]))
                if box[1] > 0.95:
                    label = '{}: {:.2f}'.format(classes[int(box[0]) - 1], box[1])
                    color = colors[classes[int(box[0]) - 1]]
                    cv2.rectangle(frame, pt_min, pt_max, color, 1)

            # save the frame
            sec_count += 1
            if sec_count == 60:
                min_count += 1
                sec_count = 0
            pred_path = '{}/{}_{}.png'.format(args.opdir, min_count, sec_count)
            logging.debug('Saving Frame At: {}'.format(pred_path))
            cv2.imwrite(pred_path, frame)

    video.release()