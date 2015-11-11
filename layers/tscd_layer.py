
import numpy

import theano
import theano.tensor as T
from convolutional_layer import ConvLayer
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


class TSCDLayer(object):
    """Class of text structure component detector layer"""

    def __init__(self, rng, input, filter_size, filter_shapes, image_shape):
        """
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_size: int
        :param filter_size: number of text structure component detector in this layer

        :type filter_shapes: tuple or list of length 3
        :param filter_shapes: (number of filters, filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)


        """

        assert len(filter_shapes) == filter_size
        self.input = input

        tscd_list = []
        self.params = []
        self.output_list = []
        for i in range filter_size:
            tscd = conv(rng, input=input, image_shape=image_shape,
                filter_shape=(filter_shapes[i][0], image_shape[1],
                              filter_shapes[i][1], filter_shapes[i][2]))
            tscd_list.append(tscd)
        for i in range filter_size:
            self.output_list.append(tscd._list[i].output)
            self.params = self.params + tscd._list[i]params


class TSCD_downsample(object):
    """downsampling for each detector in a TSCD layer"""

    def __init__(self, tscd, detector_num, poolsize):
        """

        :type tscd: TSCDLayer

        :type detector_num: int
        :param detector_num: number of text structure component detector in this layer
        
        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)


        """

        assert len(tscd.tscd_output_list) == detector_num
        
        self.output_list = []
        
        for i in range filter_size:
            self.output_list.append(downsample.max_pool_2d(
                input=tscd._list[i].output, ds=poolsize, ignore_border=True))
        



        
