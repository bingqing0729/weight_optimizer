import tensorflow as tf

init = tf.global_variables_initializer()
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    
    q = tf.Variable(tf.random_normal([100,5,50]))
    w = tf.Variable(tf.random_normal([100,1,50]))
    idx = tf.Variable(tf.random_normal([100,5,1]))

    test1 = tf.multiply(q,w)
    test2 = tf.reduce_sum(test1,1)
    test3 = tf.divide(test1,test2)
    test4 = tf.sum(test1,test3)