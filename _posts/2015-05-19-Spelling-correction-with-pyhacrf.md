---
layout: post
title: Spelling correction with pyhacrf
---

In this post I'll describe how to generate suggestions for 
incorrectly spelled words using the [`pyhacrf`]
(https://github.com/dirko/pyhacrf) package.
`pyhacrf` implements the Hidden Alignment Conditional
Random Field (HACRF) model in python with a `sklearn`-like
interface. 

## Learnable edit distance

One approach to spelling correction is to find the 'closest' 
dictionary word to our incorrectly spelled word. A common way
to define *distance* between strings is the 
[Levenshtein](http://en.wikipedia.org/wiki/Levenshtein_distance) distance.

A drawback of the Levenshtein distance is that all edit operations
have the same weight. For example, say we have the incorrect token:
`kat` and three candidates corrections: `hat`, `mat`, and `cat`.
All three corrections have the Levenshtein distance of 1 from `kat`
and we cannot further choose between the candidates.

A solution is to generalise the Levenshtein distance. Two possible
generalisations are:

- [*Weighted finite state transducers (WFSTs)*]
(http://en.wikipedia.org/wiki/Finite_state_transducer) 
The Levenshtein distance is a special case of a WFST 
where every transition has a weight of 1. Different 
transitions can have different weights, 
however. That allows the difference between `kat` and `cat` to
be 0.3 while `kat` to `hat` is 1.0 for example. These weights
can be learned from examples.

- [*Hidden alignment conditional random field (HACRF)*]
(http://people.cs.umass.edu/~mccallum/papers/crfstredit-uai05.pdf)
(or see my [thesis](http://scholar.sun.ac.za/handle/10019.1/96064)
on the use of the model for noisy text normalization)
This model can classify pairs as *matching* or *non-matching*. For example it
can classify the pair `(kat, cat)` as being a match and
`(kat, hat)` as a mismatch directly. It can also give probabilities
for these predictions. We will use the probability of a match as
an 'edit distance'.

## Training data

Let's use [Fei Liu's](http://www.cs.cmu.edu/~feiliu/) wonderful 
collection of Tweet misspelling [examples](http://www.hlt.utdallas.edu/~yangl/data/Text_Norm_Data_Release_Fei_Liu/).

Here are the first few examples of matching word pairs:

```python
[('0kkay', 'okay'),
 ('0n', 'on'),
 ('0neee', 'one'),
 ('0r', 'or'),
 ('1s', 'once')]
```

Because our model also needs examples of pairs that does not match,
we generate non-matching pairs by randomly combining incorrectly
spelled words and correctly spelled words:

```python
[('sausge', 'uses'),
 ('finnally', 'mummy'),
 ('kall', 'hit'),
 ('backkk', 'transfered'),
 ('bihg', 'morning')]
```

In the end we thus have 3974 positive examples and 3974 negative examples.

## Feature extraction

Now that we have training examples we have to transform the string 
pairs into *feature vectors*. The HACRF model assigns probability
to different alignments of the two strings by assigning *energy*
to positions in the lattice that summarise the different alignments.

For the example pair `('1s', 'once')` we can visualize all 
alignments as:

```
s . . . .        s (1, 0) (1, 1) (1, 2) (1, 3)
1 . . . .   or   1 (0, 0) (0, 1) (0, 2) (0, 3)
  o n c e            o      n      c      e
```

The model associates energy to each lattice position and lattice
transition by taking the inner product between a feature vector and
parameters:

\\[
\Phi _ {i,j}(\mathbf{x} _ {i,j},\mathbf{\lambda}) =
\exp \\{ \mathbf{x} _ {i,j} \mathbf{\lambda}^T \\}
\\]

There is thus a feature vector \\(\mathbf{x} _ {i,j} \\) for
each lattice position \\( {i,j} \\). We can stack these vectors
together into a \\( (I \times J \times K ) \\) tensor where
\\( I \\) is the length of the first string, \\( J \\) is the
length of the second string, and \\( K \\) is the length of
each feature vector. This tensor is nicely stored in a numpy
`ndarray` of shape `(I, J, K)`.

So feature extraction starts with a list of string pairs 
and outputs a list of `ndarray`s:

```python
fe = StringPairFeatureExtractor(match=True, numeric=True)
x = fe.fit_transform(x_raw)
```

where `x` now contains arrays of shapes:

```python
>>> [x_n.shape for x_n in x]
[(4, 4, 3),
 (8, 5, 3),
 (11, 10, 3),
    ...
 (7, 7, 3),
 (5, 10, 3)]
```
Each example is now an `ndarray` that is filled 
with ones and zeros:

```python
>>> x[0]
array([[[ 1.,  0.,  0.],
        [ 1.,  1.,  0.],
        [ 1.,  0.,  0.],
        [ 1.,  1.,  0.]],

       [[ 1.,  0.,  0.],
        [ 1.,  0.,  0.],
        [ 1.,  1.,  0.],
        [ 1.,  0.,  0.]],

       [[ 1.,  0.,  0.],
        [ 1.,  0.,  0.],
        [ 1.,  0.,  0.],
        [ 1.,  1.,  0.]],

       [[ 1.,  0.,  0.],
        [ 1.,  0.,  0.],
        [ 1.,  0.,  0.],
        [ 1.,  1.,  0.]]])
```

The feature vector for each lattice position
has three dimensions (if we were to unstack them). 

- The first dimension is always 1 - this is the bias
or offset feature. 
- The second is 1 when the characters of the two input
strings in that lattice position is equal and 0 otherwise.
- The third element is 1 when the characters are equal and both
are also numerical characters and zero otherwise.

## Learning

The training labels are just a list of strings:

```python
>>> y
['match', 'mismatch', ... 'mismatch']
```
We can now learn a model:

```python
>>> from sklearn.cross_validation import train_test_split
>>> x_train, x_test, y_train, y_test = \
      train_test_split(x, y, test_size=0.2, random_state=42)
>>> model = Hacrf()
>>> model.fit(x_train, y_train, verbosity=5)
Iteration  Log-likelihood |gradient|
         0     -138.6      528.7
         5     -44.44      70.03
        10     -37.29      150.9
                ...
       410     -19.51   0.007586
       415     -19.51   0.002551
```

## Evaluation

To evaluate how well it has generalized, let's compute the
confusion matrices on the training and testing sets.

```python
>>> from sklearn.metrics import confusion_matrix
>>> from sklearn.metrics import accuracy_score
>>> pr = m.predict(x_train)
>>> accuracy_score(y_train, pr)
0.96
>>> confusion_matrix(y_train, pr)
[[96  5]
 [ 3 96]]
```

And the final performance on the test set is:

```python
>>> pr = m.predict(x_test)
>>> accuracy_score(y_test, pr)
0.951260504202
>>> confusion_matrix(y_test, pr)
[[571  39]
 [ 19 561]]
``` 

It looks like the model has generalized well 
(the test and training scores are similar) and can differentiate
between random word pairs and matching word pairs. Can it, however,
give reasonable scores that can be used as a distance measure?

To see how it ranks candidate corrections, let's use it to score
candidate correct tokens from a thousand random words.

```python
# Construct list of candidate pairs
incorrect = 'ses'
candidates = set(candidate for _, candidate in x_raw[:1000])
test_pairs = [(incorrect, candidate) for candidate in candidates]

# Extract features
x_test = fe.transform(test_pairs)

# Score
pr = m.predict_proba(x_test)

# Display
candidate_scores = zip(pr, test_pairs)
candidate_scores = sorted(candidate_scores, key=lambda x: -x[0][0])
print candidate_scores[:10]
```

which produces

```
[(array([ 0.99492903,  0.00507097]), ('ses', 'she')),
 (array([ 0.99460332,  0.00539668]), ('ses', 'seems')),
 (array([ 0.98942973,  0.01057027]), ('ses', 'sheesh')), 
 (array([ 0.98908823,  0.01091177]), ('ses', 'sake')), 
 (array([ 0.98908823,  0.01091177]), ('ses', 'some')), 
 (array([ 0.98788131,  0.01211869]), ('ses', 'singers')), 
 (array([ 0.97803790,  0.02196210]), ('ses', 'send')), 
 (array([ 0.97687055,  0.02312945]), ('ses', 'shine')), 
 (array([ 0.97687055,  0.02312945]), ('ses', 'space')), 
 (array([ 0.97633783,  0.02366217]), ('ses', 'says'))]
```

The correct token is 'uses', which it doesn't get in the top ten
candidates. Here are, for a few other words, 
the top ten candidates and the probability 
of them being a match:

<table>
<tr><td colspan="2">mummmyyy</td><td colspan="2">m0rning</td><td colspan="2">mannz</td><td colspan="2">baak</td><td colspan="2">bekause</td></tr>
<tr><td>mummy</td><td>0.98</td><td>morning</td><td>1.00</td><td>man</td><td>0.99</td><td>back</td><td>0.99</td><td>because</td><td>1.00</td></tr>
<tr><td>mommy</td><td>0.65</td><td>might</td><td>0.89</td><td>mean</td><td>0.99</td><td>break</td><td>0.98</td><td>breakfast</td><td>0.93</td></tr>
<tr><td>may</td><td>0.11</td><td>missing</td><td>0.89</td><td>meant</td><td>0.94</td><td>balkan</td><td>0.97</td><td>been</td><td>0.91</td></tr>
<tr><td>much</td><td>0.10</td><td>mine</td><td>0.89</td><td>may</td><td>0.92</td><td>bad</td><td>0.96</td><td>bike</td><td>0.88</td></tr>
<tr><td>me</td><td>0.05</td><td>marketing</td><td>0.86</td><td>mad</td><td>0.92</td><td>bike</td><td>0.94</td><td>bias</td><td>0.87</td></tr>
<tr><td>mouth</td><td>0.05</td><td>turning</td><td>0.83</td><td>mais</td><td>0.74</td><td>baby</td><td>0.89</td><td>be</td><td>0.82</td></tr>
<tr><td>sum</td><td>0.05</td><td>burning</td><td>0.83</td><td>make</td><td>0.74</td><td>bias</td><td>0.89</td><td>babes</td><td>0.79</td></tr>
<tr><td>maybe</td><td>0.01</td><td>fronting</td><td>0.81</td><td>made</td><td>0.74</td><td>be</td><td>0.88</td><td>break</td><td>0.76</td></tr>
<tr><td>mad</td><td>0.01</td><td>printing</td><td>0.80</td><td>me</td><td>0.71</td><td>bastards</td><td>0.86</td><td>bread</td><td>0.74</td></tr>
<tr><td>man</td><td>0.01</td><td>training</td><td>0.78</td><td>my</td><td>0.71</td><td>bread</td><td>0.78</td><td>please</td><td>0.69</td></tr>
</table>

## Conclusion

The model gives pretty good spelling recommendations, although the 
examples we looked at are not particularly difficult. In a next post
I'll discuss how to

- add character features (the current model still cannot 
  differenciate between `hat` and `cat`),
- regularize,
- use a different optimizer.
