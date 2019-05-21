# batch generator class

import sklearn
from copy import deepcopy
import os
import cv2
import tensorflow as tf
import pandas as pd
import numpy as np
import random
from PIL import Image

class SSDLoss(object):
    def __init__(self,
                 neg_pos_ratio = 3,
                 n_neg_min = 0,
                 alpha = 1.0):
        '''
        The SSD loss, see https://arxiv.org/abs/1512.02325.
        Args:
            neg_pos_ratio (int): the maximum ratio of negative to positive samples
            n_neg_min (int): the minimum number of negative boxes
            alpha (float): factor to weigh the localization
        '''
        self.neg_pos_ratio = tf.constant(neg_pos_ratio)
        self.n_neg_min = tf.constant(n_neg_min)
        self.alpha = tf.constant(alpha)

    def log_loss(self, y_true, y_pred):
        # get log loss
        y_pred = tf.maximum(y_pred, 1e-15)
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis = -1)
        return log_loss

    def smooth_l1_loss(self, y_true, y_pred):
        # get l1 loss
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis = -1)

    def compute_loss(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        n_boxes = tf.shape(y_pred)[1]

        # 1. Compute losses for classes and box predictions
        classification_loss = tf.to_float(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12]))
        localization_loss = tf.to_float(self.smooth_l1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8]))

        # 2. Compute classification loss and positive and negative targets
        negatives = y_true[:,:,0]
        positives = tf.to_float(tf.reduce_max(y_true[:,:,1:-12], axis = -1))
        n_pos = tf.reduce_sum(positives)
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis = -1)
        neg_class_loss_all = classification_loss * negatives
        n_neg_loss = tf.count_nonzero(neg_class_loss_all, dtype = tf.int32)
        n_neg_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_pos), self.n_neg_min),n_neg_loss)

        def _f1():
            return tf.zeros([batch_size])

        def _f2():
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])
            values, indices = tf.nn.top_k(neg_class_loss_all_1D, n_neg_keep, False)
            neg_keep = tf.scatter_nd(tf.expand_dims(indices, axis = 1),
                                     updates = tf.ones_like(indices, dtype = tf.int32),
                                     shape = tf.shape(neg_class_loss_all_1D))
            neg_keep = tf.to_float(tf.reshape(neg_keep, [batch_size, n_boxes]))
            neg_class_loss = tf.reduce_sum(classification_loss * neg_keep, axis = -1)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_loss, tf.constant(0)), _f1, _f2)
        class_loss = pos_class_loss + neg_class_loss
        loc_loss = tf.reduce_sum(localization_loss * positives, axis = -1)
        total_loss = (class_loss + self.alpha*loc_loss) / tf.maximum(1.0, n_pos)
        return total_loss

def _translate(image, horizontal=(0, 40), vertical=(0, 10)):
    '''
    Randomly translate the input image horizontally and vertically.

    Arguments:
        image (array-like): The image to be translated.
        horizontal (int tuple, optinal): A 2-tuple `(min, max)` with the minimum
            and maximum horizontal translation. A random translation value will
            be picked from a uniform distribution over [min, max].
        vertical (int tuple, optional): Analog to `horizontal`.

    Returns:
        The translated image and the horzontal and vertical shift values.
    '''
    rows, cols, ch = image.shape

    x = np.random.randint(horizontal[0], horizontal[1] + 1)
    y = np.random.randint(vertical[0], vertical[1] + 1)
    x_shift = random.choice([-x, x])
    y_shift = random.choice([-y, y])

    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(image, M, (cols, rows)), x_shift, y_shift


def _flip(image, orientation='horizontal'):
    '''
    Flip the input image horizontally or vertically.
    '''
    if orientation == 'horizontal':
        return cv2.flip(image, 1)
    else:
        return cv2.flip(image, 0)


def _scale(image, min=0.9, max=1.1):
    '''
    Scale the input image by a random factor picked from a uniform distribution
    over [min, max].

    Returns:
        The scaled image, the associated warp matrix, and the scaling value.
    '''

    rows, cols, ch = image.shape

    # Randomly select a scaling factor from the range passed.
    scale = np.random.uniform(min, max)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
    return cv2.warpAffine(image, M, (cols, rows)), M, scale


def _brightness(image, min=0.5, max=2.0):
    '''
    Randomly change the brightness of the input image.

    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min, max)

    # To protect against overflow: Calculate a mask for all pixels
    # where adjustment of the brightness would exceed the maximum
    # brightness value and set the value to the maximum at those pixels.
    mask = hsv[:, :, 2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:, :, 2] * random_br)
    hsv[:, :, 2] = v_channel

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _histogram_eq(image):
    '''
    Perform histogram equalization on the input image.

    See https://en.wikipedia.org/wiki/Histogram_equalization.
    '''

    image1 = np.copy(image)

    image1[:, :, 0] = cv2.equalizeHist(image1[:, :, 0])
    image1[:, :, 1] = cv2.equalizeHist(image1[:, :, 1])
    image1[:, :, 2] = cv2.equalizeHist(image1[:, :, 2])

    return image1

class BatchGenerator(object):
    def __init__(self,
                 labels_path = None,
                 input_format = None, 
                 include_classes = 'all',
                 box_output_format = ['class_id', 'xmin', 'xmax', 'ymin', 'ymax']):
        self.labels_path = labels_path
        self.input_format = input_format
        self.include_classes = include_classes
        self.box_output_format = box_output_format
        
        self.filenames = []
        self.labels = [] # (xmin, xmax, ymin, ymax, class_id)

        # make the parse_csv function redundant
        data = []
        df = pd.read_csv(self.labels_path)
        files = df['files'].values
        classes = df['class'].values
        xmin = df['x'].values
        ymin = df['y'].values
        xmax = xmin + df['w'].values
        ymax = ymin + df['h'].values
        for i in range(len(df)):
            tup = tuple([files[i], classes[i], xmin[i], ymin[i], xmax[i], ymax[i]])
            data.append(tup)
        data = sorted(data) # The data needs to be sorted, otherwise the next step won't give the correct result

        # convert data to training format
        current_file = data[0][0]
        current_labels = []
        add_to_dataset = False
        for i, box in enumerate(data):
            if box[0] == current_file:
                current_labels.append(box[1:])
                if i == len(data)-1:
                    # If this is the last line of the CSV file
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
            
            else:
                self.labels.append(np.stack(current_labels, axis=0))
                self.filenames.append('.' + current_file)
                current_labels = []
                current_file = box[0]
                current_labels.append(box[1:])
                if i == len(data)-1:
                    # If this is the last line of the CSV file
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append('.' + current_file)

    def generate(self,
                 batch_size = 32,
                 shuffle = True,
                 train = True,
                 ssd_box_encoder = None,
                 returns = {'processed_images', 'encoded_labels'},
                 equalize = False,
                 brightness = False,
                 flip = False,
                 translate = False,
                 scale = False,
                 include_thresh = 0.3):
        '''
        Args:
            batch_size (int): mini-batch size
            shuffle (bool): To shuffle the data or not
            train (bool): To train the model or not
            ssd_box_encoder (object)
            returns (dict): Dictionary of the items to return
            equalize (bool): To perform histogram equalisation
            brightness (tuple): To scale the factor of brightness
            flip (float): probability of performing flipping
            translate (tuple): ((min, max), (min, max), prob) tuple
            scale (tuple): (min, max, prob) tuple
            include_thresh (float): box inclusion threshold
        '''
        if shuffle:
            if self.filenames and self.labels:
                self.filenames, self.labels = sklearn.utils.shuffle(self.filenames, self.labels)
            elif self.filenames:
                self.filenames = sklearn.utils.shuffle(self.filenames)

        # Find out the indices of the box coordinates in the label data
        xmin = self.box_output_format.index('xmin')
        ymin = self.box_output_format.index('ymin')
        xmax = self.box_output_format.index('xmax')
        ymax = self.box_output_format.index('ymax')
        ios = np.amin([xmin, ymin, xmax, ymax]) # Index offset, we need this for the inverse coordinate transform indices.
        
        current = 0
        while True:
            # main yield-ing loop
            batch_x, batch_y = [], []
            if current > len(self.filenames):
                current = 0 # reset
            batch_filenames = self.filenames[current:current+batch_size]

            if self.labels:
                batch_y = deepcopy(self.labels[current:current+batch_size])
            else:
                batch_y = None
            
            for i,fname in enumerate(batch_filenames):
                # iterate, normalise images
                img = Image.open(fname)
                img = np.array(img)
                img_height, img_width, img_chan = img.shape

                if equalize:
                    # perform histogram equalisation
                    img = _histogram_eq(img)

                if brightness:
                    p = np.random.uniform(0,1)
                    if p >= (1-brightness[2]):
                        img = _brightness(img, min = brightness[0], max = brightness[1])

                if flip:
                    p = np.random.uniform(0,1)
                    if p >= (1-flip):
                        img = _flip(img)
                        if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                            # xmin and xmax are swapped when mirrored
                            batch_y[i][:,[xmin,xmax]] = img_width - batch_y[i][:,[xmax,xmin]]

                if translate:
                    p = np.random.uniform(0,1)
                    if p >= (1-translate[2]):
                        # Translate the image and return the shift values so that we can adjust the labels
                        img, xshift, yshift = _translate(img, translate[0], translate[1])
                        if not ((batch_y is None) or (len(batch_y[i]) == 0)):
                            before_limiting = deepcopy(batch_y[i])
                            x_coords = batch_y[i][:,[xmin,xmax]]
                            x_coords[x_coords >= img_width] = img_width - 1
                            x_coords[x_coords < 0] = 0
                            batch_y[i][:,[xmin,xmax]] = x_coords
                            y_coords = batch_y[i][:,[ymin,ymax]]
                            y_coords[y_coords >= img_height] = img_height - 1
                            y_coords[y_coords < 0] = 0
                            batch_y[i][:,[ymin,ymax]] = y_coords
                            # when objects get pushed beyond body during transformation, they are not useful, we remove those boxes
                            before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * \
                                          (before_limiting[:,ymax] - before_limiting[:,ymin])
                            after_area = (batch_y[i][:,xmax] - batch_y[i][:,xmin]) * \
                                         (batch_y[i][:,ymax] - batch_y[i][:,ymin])
                            batch_y[i] = batch_y[i][after_area >= include_thresh * before_area]

                if scale:
                    p = np.random.uniform(0, 1)
                    if p >= (1 - scale[2]):
                        img, M, scale_factor = _scale(img, scale[0], scale[1])
                        if len(batch_y) != 0:
                            # Transform two opposite corner points of the rectangular boxes using the transformation matrix `M`
                            toplefts = np.array([batch_y[i][:,xmin], batch_y[i][:,ymin], np.ones(batch_y[i].shape[0])])
                            bottomrights = np.array([batch_y[i][:,xmax], batch_y[i][:,ymax], np.ones(batch_y[i].shape[0])])
                            new_toplefts = (np.dot(M, toplefts)).T
                            new_bottomrights = (np.dot(M, bottomrights)).T
                            batch_y[i][:,[xmin,ymin]] = new_toplefts.astype(np.int)
                            batch_y[i][:,[xmax,ymax]] = new_bottomrights.astype(np.int)

                            if scale_factor > 1:
                                before_limiting = deepcopy(batch_y[i])
                                x_coords = batch_y[i][:,[xmin,xmax]]
                                x_coords[x_coords >= img_width] = img_width - 1
                                x_coords[x_coords < 0] = 0
                                batch_y[i][:,[xmin,xmax]] = x_coords
                                y_coords = batch_y[i][:,[ymin,ymax]]
                                y_coords[y_coords >= img_height] = img_height - 1
                                y_coords[y_coords < 0] = 0
                                batch_y[i][:,[ymin,ymax]] = y_coords
                                # when objects get pushed beyond body during transformation, they are not useful, we remove those boxes
                                before_area = (before_limiting[:,xmax] - before_limiting[:,xmin]) * \
                                              (before_limiting[:,ymax] - before_limiting[:,ymin])
                                after_area = (batch_y[i][:,xmax] - batch_y[i][:,xmin]) * \
                                              (batch_y[i][:,ymax] - batch_y[i][:,ymin])
                                batch_y[i][after_area >= include_thresh * before_area]
                
                batch_x.append(img)

            # convert to numpy array for quicker handling
            batch_x = np.array(batch_x).astype(np.float32)
            batch_x -= 127.5
            batch_x /= 127.5

            # Array of shape `(batch_size, 4, 2)`, where the last axis contains an additive and a
            # multiplicative scalar transformation constant.
            batch_inverse_coord_transform = np.array([[[0, 1]] * 4] * batch_size, dtype = np.float32)

            current += batch_size # update

            if train:
                if not ssd_box_encoder:
                    raise ValueError('`ssd_box_encoder` cannot be `None` in training mode.')
                else:
                    # Encode the labels into the `y_true` tensor that the SSD loss function needs.
                    batch_y_true = ssd_box_encoder.encode_y(batch_y, diagnostics = False)

            ret = [batch_x]
            if train: ret.append(batch_y_true)
            if 'processed_labels' in returns and not batch_y is None: ret.append(batch_y)
            if 'filenames' in returns: ret.append(batch_filenames)

            yield ret

    def get_n_samples(self):
        # get the number of training samples
        return len(self.filenames)



            

        