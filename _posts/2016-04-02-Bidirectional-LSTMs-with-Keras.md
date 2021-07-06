---
layout: post
title: Masked bidirectional LSTMs with Keras
---

[Bidirectional recurrent neural networks](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks) 
(BiRNNs) enable us to classify
each element in a sequence while using information from that element's past and
future.
[Keras](https://keras.io/) provides a high level interface to 
[Theano](https://github.com/Theano/Theano) and 
[TensorFlow](https://github.com/tensorflow/tensorflow).
In this post I'll describe how to implement 
BiRNNs with Keras without using `go_backwards`
(there are different ways to skin a cat).

## Sequence tagging

Sequence tagging falls in the 
[many-to-many](http://karpathy.github.io/2015/05/21/rnn-effectiveness/#Recurrent) 
paradigm where there are as many labels as inputs.

![Many-to-many](/images/2016-04-02-Bidirectional-LSTMs-with-Keras/many_to_many.png "Many-to-many")

Examples of traditional NLP sequence tagging tasks include 
[chunking](http://www.cnts.ua.ac.be/conll2000/chunking/) and 
[named entity recognition](http://www.cnts.ua.ac.be/conll2003/ner/) (example above). 

## Sequence tagging with unidirectional LSTM
Although you can do a straight implementation of the diagram above 
(by feeding examples to the network 
[one by one](https://github.com/fchollet/keras/issues/40)), you would 
immediately find that it is much to slow to be useful. To speed it up
we need to vectorise the vectoriseable. This means that examples must be fed
into the network in mini-batches.

The problem with this is of course that sequences are different lengths 
- which is exactly why we are using RNNs in the first place---we want to be
able to handle any length input.

There are a few ways to get variable length sequences *and* vectorisation.
The one that seems to be the most popular is to fill in the sequences into
an array block of size `(N, maxlen, D)`, where `maxlen` is the length of the
longest sequence in the set, and then zero-pad the rest. 
Then you enable masking on the sequence layer (LSTM/GRU/etc), which disables
recurrent computation when the input is zero.

![Input](/images/2016-04-02-Bidirectional-LSTMs-with-Keras/input_block.png "Input")

Our model now looks like this (see this 
[gist](https://gist.github.com/dirko/375397bc942d134a3c82d0dd514f3fea)
 for the full preprocessing and training code):

```python
max_features = len(word2ind)
embedding_size = 128
hidden_size = 32
out_size = len(label2ind) + 1

model = Sequential()
model.add(Embedding(input_dim=max_features,
                    output_dim=embedding_size,
                    input_length=maxlen,
                    mask_zero=True))
model.add(LSTM(hidden_size, return_sequences=True))  
model.add(TimeDistributedDense(out_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
```

There are also subtleties when you want to break up long sequences into
consecutive mini-batches (the RNN state must not be reset), but that is a 
story for another day (See Keras' 
[stateful RNNs](https://keras.io/layers/recurrent/#recurrent)).

## Looking both ways before crossing

With the model above we get to predict the current element's label with
information from the present and the past. But what if the next element 
(or the last one for that matter) helps to predict the current label?

For example if the current word is `new` then it is probably not a named 
entity, except if the next word is `york`.

Once again there are different ways to do this---you can do a pass through the
sequence before labelling, or you can have RNNs going forward
and backwards simultaneously as with a BiRNN. 

## But what about padding?

There has been a bit of a debate about how to implement BiRNNs with Keras
(See [1](https://github.com/fchollet/keras/issues/418), 
[2](https://github.com/fchollet/keras/pull/1282), 
[3](https://github.com/fchollet/keras/issues/1629),
[4](https://github.com/fchollet/keras/issues/2838),
and [5](https://github.com/fchollet/keras/issues/2536)),
 and to be honest I haven't followed everything.
It seems to me that there are two ways to think about it:

- Use a single input and then set the `go_backwards` flag 
  (which is passed through to Theano/TF) on one of the passes. 
  This implementation depends on [how exactly masking works with
  `go_backward`](https://github.com/fchollet/keras/issues/1057)
  (which still confuses me---maybe someone can explain to 
  me to the current state of `go_backward` with masking?). 
- The user must supply both normal and reversed inputs. The disadvantage here
  is that our input data size is doubled.

In my case I like to play with smaller datasets anyways so the second option
looks much more understandable.

![Input](/images/2016-04-02-Bidirectional-LSTMs-with-Keras/birnn.png "Input") 

## Reverse

The next crucial building block is a way to reverse sequences, *and also 
their masks*. [One way](https://github.com/fchollet/keras/issues/2076)
to reverse sequences in Keras is with 
a Lambda layer that wraps `x[:,::-1,:]` on the input tensor.
Unfortunately I couldn't find a way in straight Keras 
that will also reverse the mask, but 
[`@braingineer`](https://github.com/braingineer) created the perfect
[custom lambda layer](https://gist.github.com/braingineer/b64ca35223c7782667984d34ddb7a7fa)
that allows us to manipulate the mask with an arbitrary 
function. 

Using the custom lambda:

```python
from keras.backend import tf
from lambdawithmask import Lambda as MaskLambda

def reverse_func(x, mask=None):
    # For theano back-end:
    # return x[:,::-1,:]

    # For tensorflow back-end:
    return tf.reverse(x, [False, True, False])

reverse = MaskLambda(function=reverse_func, mask_function=reverse_func) 
```
(The tensorflow back-end doesn't support `[:, ::-1, :]` giving
`NotImplementedError: Steps other than 1 are not currently supported`, 
but luckily it works with `.reverse()`.)

## Merge

Merge layers didn't support masking until recently 
(See [1](https://github.com/fchollet/keras/issues/2393), 
[2](https://github.com/fchollet/keras/pull/2413)),
but works with `Keras==1.0.6`.

## Model

Using the good old `Sequential` setup our bidirectional RNN now looks like:

```python
model_forward = Sequential()
model_forward.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))
model_forward.add(LSTM(hidden_size, return_sequences=True))  

model_backward = Sequential()
model_backward.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))
model_backward.add(LSTM(hidden_size, return_sequences=True))  
def reverse_func(x, mask=None):
    return tf.reverse(x, [False, True, False])
model_backward.add(MaskLambda(function=reverse_func, mask_function=reverse_func))

model = Sequential()
model.add(Merge([model_forward, model_backward], mode='concat'))
model.add(TimeDistributedDense(out_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
```
([Here](https://gist.github.com/dirko/1d596ca757a541da96ac3caa6f291229) 
is the gist of everything.)

Training is done by simply passing in the two versions of the input
sequence to:

```python
model.fit([X_train_f, X_train_b], y_train, batch_size=batch_size)`
```

## Conclusion

Some other things that I want to explore:

 - getting the hang of `go_backwards`---it might simplify the model by making it
 'stackable' (make it behave like a layer so we can stack a few together to 
 create a deep BiRNN). 
 It would probably need a lambda layer before the backwards RNN to
 align the mask. Come to think of it, we can make the current model stackable
 like that.
 - use the functional API to share the embeddings between the 
 forward and backward parts.

Next I'll post the results of some experiments on the 
[CoNLL named entity](http://www.cnts.ua.ac.be/conll2003/ner/) task.

*Thanks for all @fchollet and the multitude of people working on Keras! 
(and Theano, and Tensorflow, and Python---opensource is amazing)*

