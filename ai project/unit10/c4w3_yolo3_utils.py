import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, \
    Input, ZeroPadding2D, LeakyReLU, UpSampling2D
import cv2
import time


########################### load config file yolov3.cfg #########################
def parse_cfg(cfgfile):
    with open(cfgfile, 'r') as file:
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    holder = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()
    blocks.append(holder)
    return blocks
def YOLOv3Net(cfgfile, model_size, num_classes):
    blocks = parse_cfg(cfgfile)
    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0
    inputs = input_image = Input(shape=model_size)
    inputs = inputs / 255.0
    for i, block in enumerate(blocks[1:]):
        # If it is a convolutional layer
        if (block["type"] == "convolutional"):
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])
            if strides > 1:
                inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
            inputs = Conv2D(filters,
                            kernel_size,
                            strides=strides,
                            padding='valid' if strides > 1 else 'same',
                            name='conv_' + str(i),
                            use_bias=False if ("batch_normalize" in block) else True)(inputs)
            if "batch_normalize" in block:
                inputs = BatchNormalization(name='bnorm_' + str(i))(inputs)
            #if activation == "leaky":
                inputs = LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)
        elif (block["type"] == "upsample"):
            stride = int(block["stride"])
            inputs = UpSampling2D(stride)(inputs)
        # If it is a route layer
        elif (block["type"] == "route"):
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0])
            if len(block["layers"]) > 1:
                end = int(block["layers"][1]) - i
                filters = output_filters[i + start] + output_filters[end]  # Index negatif :end - index
                inputs = tf.concat([outputs[i + start], outputs[i + end]], axis=-1)
            else:
                filters = output_filters[i + start]
                inputs = outputs[i + start]
        elif block["type"] == "shortcut":
            from_ = int(block["from"])
            inputs = outputs[i - 1] + outputs[i + from_]
        # Yolo detection layer
        elif block["type"] == "yolo":
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            n_anchors = len(anchors)
            out_shape = inputs.get_shape().as_list()
            inputs = tf.reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], \
           5 + num_classes])
            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5:num_classes + 5]
            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)
            anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)
            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)
            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, n_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])
            strides = (input_image.shape[1] // out_shape[1], \
                       input_image.shape[2] // out_shape[2])
            box_centers = (box_centers + cxy) * strides
            prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1
        outputs[i] = inputs
        output_filters.append(filters)
    model = Model(input_image, out_pred)
    model.summary()
    return model

########################### load weight file yolov3.weights #########################

def load_weights(model,cfgfile,weightfile):
    # Open the weights file
    fp = open(weightfile, "rb")
    # Skip 5 header values
    np.fromfile(fp, dtype=np.int32, count=5)
    # The rest of the values are the weights
    blocks = parse_cfg(cfgfile)
    for i, block in enumerate(blocks[1:]):
        if (block["type"] == "convolutional"):
            conv_layer = model.get_layer('conv_' + str(i))
            print("layer: ",i+1,conv_layer)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]
            if "batch_normalize" in block:
                norm_layer = model.get_layer('bnorm_' + str(i))
                print("layer: ",i+1,norm_layer)
                size = np.prod(norm_layer.get_weights()[0].shape)
                bn_weights = np.fromfile(fp, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            else:
                conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)
            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(
                fp, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])
            if "batch_normalize" in block:
                norm_layer.set_weights(bn_weights)
                conv_layer.set_weights([conv_weights])
            else:
                conv_layer.set_weights([conv_weights, conv_bias])
    assert len(fp.read()) == 0, 'failed to read all data'
    fp.close()


########################### load class names from file #########################
def load_class_names(file_name):
    """Returns a list of class names read from `file_name`."""
    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

########################### resize image to new_dimensions #########################
def resize_image(image, new_dimensions):
    image = tf.image.resize(image, new_dimensions)
    return image

########################### output boxes #########################
def output_boxes(inputs,model_size, max_output_size, max_output_size_per_class, 
                 iou_threshold, confidence_threshold):
    """converts the boxes into the format of (top-left-corner, bottom-right-corner), 
    followed by applying the NMS function and returning the proper bounding boxes."""

    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0
    inputs = tf.concat([top_left_x, top_left_y, bottom_right_x,
                        bottom_right_y, confidence, classes], axis=-1)
    boxes_dicts = NMS(inputs, model_size, max_output_size, 
                                      max_output_size_per_class, iou_threshold, confidence_threshold)
    return boxes_dicts

########################### draws the final boxes on the image #########################
def draw_outputs(img, boxes, objectness, classes, nums, class_names):
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes=np.array(boxes)
    for i in range(nums):
        x1y1 = tuple((boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32))
        x2y2 = tuple((boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32))
        img = cv2.rectangle(img, (x1y1), (x2y2), (255,0,0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
                          (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    return img

########################### a slightly different version of non max supression #########################
def NMS(inputs, model_size, max_output_size,
                        max_output_size_per_class, iou_threshold,
                        confidence_threshold):
    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox=bbox/model_size[0]
    scores = confs * class_probs
    boxes, scores, classes, valid_detections = \
        tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1,
                                   tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )
    return boxes, scores, classes, valid_detections
