#conv, pool, norm, conv, pool, norm, affine, dropout, affine, softmax

def model(X):
    #conv1
    with tf.variable_scope('conv1') as scope:
        filters = tf.get_variable('filters', shape=[5, 5, 3, 64])
        conv = tf.nn.conv2d(X, filters, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    pool1 = tf.layers.max_pooling2d(conv1, (3, 3), 2, padding='valid', name='pool1')
    norm1 = tf.layers.batch_normalization(pool1, name='norm1')
    #norm1 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    #conv2
    with tf.variable_scope('conv2') as scope:
        filters = tf.get_variable('filters', shape=[5, 5, 64, 64])
        conv = tf.nn.conv2d(norm1, filters, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    pool2 = tf.layers.max_pooling2d(conv2, (3, 3), 2, padding='valid', name='pool2')
    norm2 = tf.layers.batch_normalization(pool2, name='norm2')
    #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    #affine1
    with tf.variable_scope('affine1') as scope:
        #norm2 shape: 7*7*64 = 3136
        reshape = tf.reshape(norm2, [-1, 3136])
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

def loss(logits, y):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    return cross_entropy_mean

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
logits = model(X)
mean_loss = loss(logits, y)

# Optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)

run_model(logits, mean_loss, y, train_step, X_train, y_train, 1, 64, 100, 'gpu')
