import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import progressbar
import os

def train(directory_name, model, dataset, optimizer, minibatch_size, n_epochs, srng, **kwargs):
    print "training for {} epochs with {} learning rate".format(n_epochs, optimizer.learning_rate)
    num_minibatches = dataset.get_n_examples('train') / minibatch_size

    index = T.lscalar('i')
    minibatch = dataset.minibatchIindex_minibatch_size(index, minibatch_size, srng=srng, subdataset='train')

    grad, weights = model.gradIminibatch_srng(minibatch, srng, **kwargs)
    updates = optimizer.updatesIgrad_model(grad, model)

    train_step = theano.function([index], None, updates=updates)
    
    #log_marginal_likelihood_estimate = model.get_log_ws(minibatch, srng, 50)

    get_log_ws = theano.function([index], (weights))

    #pbar = progressbar.ProgressBar(maxval=num_minibatches).start()
    weights = []
    effectiveSamples = []
    effectiveSamples1 = []
    threshold = 0.01
    threshold1 = 0.1

    pbar = progressbar.ProgressBar(maxval=n_epochs*num_minibatches).start()
    for j in xrange(n_epochs):
        for i in xrange(num_minibatches):
            train_step(i)
            summand = get_log_ws(i)
            weights += summand.tolist()
	    effectiveSamples += [sum(i > threshold for i in summand.tolist())/minibatch_size]
   	    effectiveSamples1 += [sum(i > threshold1 for i in summand.tolist())/minibatch_size]
	    #print len(summand)
            pbar.update(j*num_minibatches+i)
    pbar.finish()
	
    avEffectiveSamples1 = (reduce(lambda x, y: x + y, effectiveSamples1) / float(len(effectiveSamples1)))
    avEffectiveSamples = (reduce(lambda x, y: x + y, effectiveSamples) / float(len(effectiveSamples)))
    ep = n_epochs**(1/3)+1

    with open(os.path.join(directory_name, "avEffecSamples"+str(ep)+".txt"), "w") as f:
        f.write(str(avEffectiveSamples) + ' , ' + str(avEffectiveSamples1) )

    plt.figure()
    plt.hist(weights, 100)
    plt.xlabel('normalized importance weight')
    plt.savefig(os.path.join(directory_name, "weightsDistr"+str(ep)+".jpg"))
    plt.close()
    plt.figure()
    plt.plot(effectiveSamples)
    plt.ylabel('number of effective samples')
    plt.xlabel('iteration')
    plt.savefig(os.path.join(directory_name, "effectiveSamples"+str(ep)+".jpg"))
    plt.close()
    plt.figure()
    plt.plot(effectiveSamples1)
    plt.ylabel('number of effective samples')
    plt.xlabel('iteration')
    plt.savefig(os.path.join(directory_name, "effectiveSamples_2_"+str(ep)+".jpg"))
    plt.close()
    return model

