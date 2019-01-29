def run_model(logits, mean_loss, labels, train_step, Xd, yd, epochs=1, batch_size=64, print_every=100, device='cpu'):
    def run(session, training=None, plot_losses=False):
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.argmax(logits,1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None
        
        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [mean_loss,correct_prediction,accuracy]
        if training_now:
            variables[-1] = training
        
        # counter 
        iter_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                # generate indicies for the batch
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx+batch_size]
                
                # create a feed dictionary for this batch
                feed_dict = {X: Xd[idx,:],
                             y: yd[idx],
                             is_training: training_now }
                # get batch size
                actual_batch_size = yd[idx].shape[0]
                
                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables,feed_dict=feed_dict)
                
                # aggregate performance stats
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)
                
                # print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print('Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}'\
                          .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                iter_cnt += 1
            total_correct = correct/Xd.shape[0]
            total_loss = np.sum(losses)/Xd.shape[0]
            print('Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}'\
                  .format(total_loss,total_correct,e+1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss,total_correct

    with tf.Session() as sess:
        with tf.device('/' + device + ':0'): #'/cpu:0' or '/gpu:0' 
            sess.run(tf.global_variables_initializer())
            print('Training')
            run(sess,train_step,True)
            print('Validation')
            run(sess)


