#conv, conv, conv, conv, affine, dropout, affine, softmax

def model(images):
    #conv1
    with tf.variable_scope('conv1') as scope:
        filters = tf.get_variable('filters', shape=[5, 5, 3, 64])
        conv = tf.nn.conv2d(images, filters, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #conv2
    with tf.variable_scope('conv2') as scope:
        filters = tf.get_variable('filters', shape=[5, 5, 3, 64])
        conv = tf.nn.conv2d(conv1, filters, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #conv3
    with tf.variable_scope('conv3') as scope:
        filters = tf.get_variable('filters', shape=[5, 5, 3, 64])
        conv = tf.nn.conv2d(conv2, filters, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
    #conv4
    with tf.variable_scope('conv4') as scope:
        filters = tf.get_variable('filters', shape=[5, 5, 3, 64])
        conv = tf.nn.conv2d(conv3, filters, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
    #affine1
    with tf.variable_scope('affine1') as scope:
        reshape = tf.reshape(conv4, [images.get_shape().as_list()[0], -1])
        dim1 = reshape.get_shape()[1].value
        dim2 = int(dim1 / 2)
        weights = tf.get_variable('weights', shape=[dim1, dim2])
        biases = tf.get_variable('biases', shape=[dim2], initializer=tf.constant_initializer(0.1))
        affine1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    #affine2
    with tf.variable_scope('affine2') as scope:
        dim1 = dim2
        dim2 = int(dim1 / 2)
        weights = tf.get_variable('weights', shape=[dim1, dim2])
        biases = tf.get_variable('biases', shape=[dim2], initializer=tf.constant_initializer(0.1))
        affine2 = tf.nn.relu(tf.matmul(affine1, weights) + biases, name=scope.name)
    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax') as scope:
        dim1 = dim2
        dim2 = 10
        weights = tf.get_variable('weights', shape=[dim1, dim2])
        biases = tf.get_variable('biases', shape=[dim2], initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    return softmax_linear # To feed to the loss function # logits

def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    return cross_entropy_mean

images = tf.placeholder(tf.float32, [None, 32, 32, 3])
