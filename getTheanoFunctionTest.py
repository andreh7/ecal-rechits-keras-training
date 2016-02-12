#!/usr/bin/env python

# trying to get the theano function out of the keras model

execfile("model01.py")

model = makeModel((1, 7, 23))

import theano

# create the theano function
# see e.g. https://github.com/fchollet/keras/issues/41

func = theano.function([model.layers[0].input], model.layers[-1].get_output(train=False), allow_input_downcast=True)

# print it
from theano.printing import debugprint
debugprint(func,
           # print_storage = True, # prints the values of the weights
           )
#----------
# visit the nodes
# see e.g. https://github.com/Theano/Theano/blob/master/theano/printing.py#L44
#----------


