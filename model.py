# import dep
import numpy as np
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, ELU, Reshape, Concatenate, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam

# custom
from layers import L2Norm, AnchorBoxes
from ssd_utils import SSDLoss

# custom version for ubisoft assignment
FILTERS = (32, 48, 64, 64, 48, 48, 32)
KERNELS = [(5, 5)] + [(3, 3),] * 6

class SSD7(object):
    '''
    main class to handle the model
    '''
    def __init__(self,
                 min_scale = 0.1,
                 max_scale = 0.9,
                 scales = None,
                 aspect_ratios_global = [0.5, 1.0, 2.0],
                 aspect_ratios_per_layer = None,
                 two_boxes_for_ar1 = True,
                 steps = None,
                 limit_boxes = True,
                 variances = [1.0, 1.0, 1.0, 1.0],
                 coords = 'centroids',
                 normalize_coords = False):
        '''
        Args:
            min_scale
        '''
        # primary attributes
        self.n_predictor_layers = 4
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.aspect_ratios_per_layer = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.limit_boxes = limit_boxes
        self.variances = np.array(variances)
        self.coords = coords
        self.normalize_coords = normalize_coords
        self.steps = [None,] * self.n_predictor_layers if steps is None else steps

        # secondary attributes
        self.scales = np.linspace(self.min_scale, self.max_scale, self.n_predictor_layers + 1)
        
        if not aspect_ratios_per_layer:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                n_boxes = len(aspect_ratios_global) + 1
            else:
                n_boxes = len(aspect_ratios_global)
            self.n_boxes = [n_boxes,] * 4
            self.aspect_ratio = [aspect_ratios_global,] * 4

    def load(self, path):
        self.model.load(path)

    def _network(self):
        inp = Input(shape = self.image_size)
        # @ no lambda here because we have already normalised images
        
        # add layers
        cnt = 1
        net_convs_, net_pools_ = [], []
        for num_fil, k_size in zip(FILTERS, KERNELS):
            x = inp if cnt == 1 else out
            out = Conv2D(num_fil, k_size, name = 'conv{}'.format(cnt),
                         padding = "same", kernel_initializer = 'he_normal',
                         kernel_regularizer = l2(0))(x)
            out = BatchNormalization(axis = 3, momentum = 0.99, name = 'bn{}'.format(cnt))(out)
            out = ELU(name = 'elu{}'.format(cnt))(out)
            net_convs_.append(out)
            out = MaxPooling2D((2, 2), name = 'pool{}'.format(cnt))(out)
            net_pools_.append(out)
            cnt += 1

        # next step is to add conv layers on top of base network and build on top of base network
        # we build on top of conv layers 4, 5, 6, 7
        
        net_classes_ = []
        net_boxes_ = []
        net_anchors_ = []
        self.predictor_sizes = []
        for i in range(3, 7, 1):
            # print('======')
            # classes
            c_ = Conv2D(self.n_boxes[i - 4] * self.num_classes,
                        (3, 3), strides = (1, 1), padding = 'valid',
                        name = "classes{}".format(i), kernel_initializer = 'he_normal',
                        kernel_regularizer = l2(0))(net_convs_[i])
            # print(c_)
            c_reshaped = Reshape((-1, self.num_classes),
                        name = 'classes{}_reshaped'.format(i))(c_)
            # print(c_reshaped)

            # boxes
            b_ = Conv2D(self.n_boxes[i - 4] * 4,
                        (3, 3), strides = (1, 1), padding = "valid",
                        name = "boxes{}".format(i), kernel_initializer = 'he_normal',
                        kernel_regularizer = l2(0))(net_convs_[i])
            # print(b_)
            b_reshaped = Reshape((-1, 4),
                        name = 'boxes{}_rehshaped'.format(i))(b_)
            # print(b_reshaped)

            # anchors
            a_ = AnchorBoxes(self.image_size[0], self.image_size[1],
                        this_scale = self.scales[i-4],
                        next_scale = self.scales[i-3],
                        aspect_ratios = self.aspect_ratio[i-4],
                        two_boxes_for_ar1 = self.two_boxes_for_ar1,
                        this_steps = self.steps[i-3],
                        limit_boxes = self.limit_boxes,
                        variances = self.variances,
                        coords = self.coords,
                        normalize_coords = self.normalize_coords,
                        name = 'anchors{}'.format(i))(b_)
            # print(a_)
            a_reshaped = Reshape((-1, 8),
                        name = 'anchors{}_reshaped'.format(i))(a_)
            # print(a_reshaped)
            
            # add to network lists
            net_classes_.append(c_reshaped)
            net_boxes_.append(b_reshaped)
            net_anchors_.append(a_reshaped)
            self.predictor_sizes.append(c_._keras_shape[1:3])

        # generate anchor boxes
        classes_concat = Concatenate(axis = 1, name = 'classes_concat')(net_classes_)

        # print('=====')
        # print(classes_concat)
        classes_softmax = Activation('softmax', name = 'classes_softmax')(classes_concat) # class prediction
        # print(classes_softmax)
        boxes_concat = Concatenate(axis = 1, name = 'boxes_concat')(net_boxes_)
        # print(boxes_concat)
        anchors_concat = Concatenate(axis = 1, name = 'anchors_concat')(net_anchors_)
        # print(anchors_concat)
        # output of predictions is (batch_size, n_boxes_total, n_classes + 4 + 8)
        predictions = Concatenate(axis = 2, name = 'predictions')([classes_softmax, boxes_concat, anchors_concat])
        # print(predictions)

        self.model = Model(inputs = inp, outputs = predictions)

    def build_model(self,
                    image_size,
                    num_classes,
                    ):
        '''
        Build the SSD7 model
        Args:
            image_size (tuple): (h,w,c)
            num_classes (int): number of classes
        '''
        # build network
        self.image_size = image_size
        self.num_classes = num_classes + 1 # account for the negative classes
        self._network()

        # learning steps
        adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 5e-04)
        ssd_loss = SSDLoss(neg_pos_ratio = 3, n_neg_min = 0, alpha = 1.0)
        self.model.compile(optimizer = adam, loss = ssd_loss.compute_loss)

    def train_generator(self, generator, steps_per_epoch, epochs = 10, callbacks = None):
        history = self.model.fit_generator(generator = generator,
                                           steps_per_epoch = steps_per_epoch,
                                           epochs = epochs,
                                           callbacks = callbacks)
        return history

    def predict(self, predictions):
        pass


