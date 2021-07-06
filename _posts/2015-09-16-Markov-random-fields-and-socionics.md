---
layout: post
title: Markov random fields and socionics
---

[Socionics](https://en.wikipedia.org/wiki/Socionics)
 comes across as the more 
serious and academic eastern European cousin of 
[MBTI](https://en.wikipedia.org/wiki/Myers-Briggs)
(which is much better known by English speakers).
Although many of the same criticism that apply to 
personality theories/tools
such as MBTI are also applicable to socionics, 
I find these models fascinating. Here I'll use socionics 
to set up a fun application/demonstration of Markov random fields.

## Personality profiles
These personality theories are often
discredited as unscientific or useless (with the possible exception 
of the 
[big five](https://en.wikipedia.org/wiki/Big_Five_personality_traits)---also 
see IBM's 
[showcase](https://watson-pi-demo.mybluemix.net/) of an automatic 
personality profiler that takes English text as input and spits out a 
profile of the author), but online interest seems to keeps the debate
about socionics as a science alive. 
[Ex-Socionist](http://socionist.blogspot.co.za/)'s 
blog has more on this topic.

## Socionics
Socionics (okay my understanding of one of the 
sub-understandings of *Model A*)
posits that there are 4 ways that people process information 
(originally proposed by Jung): 
(**F**)eeling, (**S**)ensing, (**T**)hinking,
and i(**N**)tuition. Each of these
*functions* are further divided into (**i**)ntroverted and (**e**)xtraverted
functions---depending on whether they are directed inward of outward.

#### Elements
We now end up with 8 information *elements*:

```python 
elements = ['Te', 'Fe', 'Se', 'Ne', 'Ti', 'Fi', 'Si', 'Ni']
```

They can be tidily poured into a 3 dimensional `ndarray`:

```python
elements = np.array(elements).reshape(2, 2, 2)

# Assign an index to each element
e2i = {element: i for i, element in enumerate(elements.flatten())}
e2ii = {elements[tuple(i)]: i for i in indices.T}

elements
> array([[['Te', 'Fe'],
         ['Se', 'Ne']],

         [['Ti', 'Fi'],
          ['Si', 'Ni']]], 
        dtype='|S2')
```

#### Types
The theory further postulates that each person uses all the elements, but
that people differ in their preferences and the roles that they assign
to each element. For example, a person's preferred conscious 
element is called the *leading* function, while the secondary element
is called the *creative* function because it assists the leading
function to produce something new. All eight elements are given
roles like these two.

The slots that the elements can take are highly constrained, however.
We only need to specify two functions to completely specify a *type*.
So personality types are often specified by their 
leading and creative functions. We are left with
sixteen valid combinations of elements:

```python
indices = np.indices((2, 2, 2)).reshape(3, 8)
types = ['{}{}'.format(elements[tuple(i)], elements[tuple(i - np.array([1, 1, j]))]) 
         for i in indices.T for j in [0, 1]]

# Assign an index to each type
t2i = {typ: index for index, typ in enumerate(types)}
i2t = {index: typ for index, typ in enumerate(types)}

types
> ['TeSi', 'TeNi', 'FeNi', 'FeSi',
   'SeTi', 'SeFi', 'NeFi', 'NeTi',
   'TiSe', 'TiNe', 'FiNe', 'FiSe',
   'SiTe', 'SiFe', 'NiFe', 'NiTe']
```

#### Relations
Socionics is also concerned with stereotypical interactions between
types. Types have elements that they are strong with and therefore 
understand in others, but also elements that they are weak with and
appreciate in others. There are, however, also weak and strong
 elements that
they don't care about or are irritated by if strongly expressed by others.

All these different combinations of type interactions are summed up
in sixteen *relation types*:

```python
relation_deltas = [
    ('duality', 'm', (-1, 0, -1), (-1, 0, -1)),
    ('activation', 'r', (-1, 0, -1), (-1, 0, -1)),
    ('semi-duality', 'm', (-1, 0, -1), (-1, 0, 0)),
    ('mirage', 'm', (-1, 0, 0), (-1, 0, -1)),
    ('mirror', 'r', (0, 0, 0,), (0, 0, 0)),
    ('identity', 'm', (0, 0, 0), (0, 0, 0)),
    ('cooperation', 'm', (0, 0, -1), (0, 0, 0)),
    ('congenerity', 'm', (0, 0, 0), (0, 0, -1)),
    ('quasi-identity', 'r', (-1, 0, 0), (-1, 0, 0)),
    ('extinguishment', 'm', (-1, 0, 0), (-1, 0, 0)),
    ('super-ego', 'm', (0, 0, -1), (0, 0, -1)),
    ('conflict', 'r', (0, 0, -1), (0, 0, -1)),
    ('requester', 'r', (-1, 0, -1), (-1, 0, 0)),
    ('request-recipient', 'r', (-1, 0, 0), (-1, 0, -1)),
    ('supervisor', 'r', (0, 0, 0), (0, 0, -1)),
    ('supervisee', 'r', (0, 0, -1), (0, 0, 0))
]
r2i = {rel: i for i, (rel, _, _, _) in enumerate(relation_deltas)}
i2r = {i: rel for i, (rel, _, _, _) in enumerate(relation_deltas)}
```
We are now ready to try and put some people into their boxes.

## Markov social network

Let's build a model that assumes each person is one of the sixteen
types but we're not sure exactly which one. The model is therefore a 
joint probability function over all types and relations for
all the persons in our network. We'll represent this model as
a Markov random field using [`pyugm`](https://github.com/dirko/pyugm).

#### Types
For each person there is thus also a distribution over types. 
This is modelled by a factor for each person that encodes our
uncertainty about their type:

```python
# Helper to contruct a potential table over types for a person.
# Potentials for each types are set to one, except those 
#     types that are passed as arguments.
def prob_type(name, *prob_typ):
    type_data = np.ones(16)
    for typ, potential in prob_typ:
        type_data[t2i[typ]] = potential
    type_factor = DiscreteFactor([('{}_type'.format(name), 16)], type_data)
    return type_factor

factors = []
people = ['Lev', 'Dil', 'Uli', 'Yre',
          'Ani', 'Itu', 'Urb', 'Rya',
          'Owa', 'Ako']

factors.append(prob_type('Lev', ('SiFe', 2.0)))
factors.append(prob_type('Dil', ('FiNe', 2.0)))
factors.append(prob_type('Itu', ('TiNe', 2.0), ('NeTi', 2.5)))
```

#### Elements
Sometimes we rather want to specify the distribution over elements
that someone likely has as leading or creative functions:

```python

# Helper to contruct a factor over the 
#     first two elements for a person.
def prob_has(name, *prob_elements):
    data = np.ones(8)
    for element, potential in prob_elements:
        data[e2i[element]] = potential
    prim_factor = DiscreteFactor([('{}_primary_function'.format(name), 8)], data)
    sec_factor = DiscreteFactor([('{}_secondary_function'.format(name), 8)], data)
    return (prim_factor, sec_factor)

factors.extend(prob_has('Uli', ('Ti', 1.5), ('Se', 1.5),
                               ('Si', 1.3), ('Te', 1.4)))
factors.extend(prob_has('Yre', ('Ne', 1.5)))
factors.extend(prob_has('Ani', ('Fe', 1.5)))
factors.extend(prob_has('Urb', ('Ne', 2.5)))
factors.extend(prob_has('Rya', ('Si', 1.2)))
factors.extend(prob_has('Owa', ('Si', 1.1), ('Ti', 1.2),
                               ('Te', 1.2)))
factors.extend(prob_has('Ako', ('Ni', 1.5), ('Ti', 1.2),
                               ('Se', 1.1), ('Si', 1.1)))
```

At the moment each person has two factors that are completely
unconnected---they don't share any variables. But we know that
the first two elements and a person's type are deterministically
tied. Let's therefore add another factor for each person to
capture the dependence between the `function` and `type` variables:

```python
element_type = np.zeros((16, 8, 8))
for typ in types:
    el1 = typ[:2]
    el2 = typ[2:]
    element_type[t2i[typ], e2i[el1], e2i[el2]] = 1
for person in people:
    variables = [('{}_type'.format(person), 16),
                 ('{}_primary_function'.format(person), 8),
                 ('{}_secondary_function'.format(person), 8)]
    factors.append(DiscreteFactor(variables, element_type))
```

#### Relations
If we strongly believe that two people's relationship is of a 
certain type, then that will constrain the types that each of them
can be. Let's therefore add a random variable `relation` between
all the pairs of persons. 

```python
# Helper to construct the relation factors.
def prob_has(name, *prob_elements):
    data = np.ones(8)
    for element, potential in prob_elements:
        data[e2i[element]] = potential
    prim_factor = DiscreteFactor([('{}_primary_function'.format(name), 8)], data)
    sec_factor = DiscreteFactor([('{}_secondary_function'.format(name), 8)],data)
    return (prim_factor, sec_factor)

relations = {
    ('Lev', 'Itu'): [('duality', 1.5), ('activation', 1.5)],
    ('Uli', 'Dil'): [('duality', 1.5), ('activation', 1.5)],
    ('Rya', 'Urb'): [('duality', 1.5), ('activation', 1.5)],
    ('Dil', 'Rya'): [('quasi-identity', 1.5), ('super-ego', 1.5),
                     ('conflict', 1.5)],
    ('Uli', 'Urb'): [('duality', 1.2), ('activation', 1.2),
                     ('semi-duality', 1.2), ('mirage', 1.2),
                     ('congenerity', 1.2)],
    ('Itu', 'Owa'): [('semi-duality', 1.2), ('requester', 1.2)],
    ('Itu', 'Uli'): [('cooperation', 1.2), ('congenerity', 1.2)],
}
for i, name1 in enumerate(people):
    for name2 in people[i + 1:]:
        factors.append(prob_relation(name1, name2, *relations.get((name1, name2), [])))
 
```

We then capture the dependency between the
types of the two persons and the relation between them with a further
deterministic factor.

```python
types_relation = np.zeros((16, 16, 16))
for i, typ1 in enumerate(types):
    el1 = typ1[:2]
    el2 = typ1[2:]
    for rel, direction, delta1, delta2 in relation_deltas:
        if direction == 'm':  
            # In some relations, the other type's leading function
            #    is defined by this type's leading function
            el2_1 = elements[tuple(e2ii[el1] + np.array(delta1))]
            el2_2 = elements[tuple(e2ii[el2] + np.array(delta2))]
        else:
            # In others, by this type's creative function
            el2_1 = elements[tuple(e2ii[el2] + np.array(delta1))]
            el2_2 = elements[tuple(e2ii[el1] + np.array(delta2))]
        typ2 = '{}{}'.format(el2_1, el2_2)
        types_relation[t2i[typ1], t2i[typ2], r2i[rel]] = 1.0

for i, name1 in enumerate(people):
    for name2 in people[i + 1:]:
        variables = [('{}_type'.format(name1), 16),
                     ('{}_type'.format(name2), 16),
                     ('{}_{}_relation'.format(name1, name2), 16)]
        factors.append(DiscreteFactor(variables, types_relation))
```

#### Inference
Now that the model is specified we can run inference and 
calculate the marginal probabilities of the different 
characters' types.

```python
model = Model(factors)
infer = LoopyBeliefUpdateInference(model)
infer.calibrate()

# Helper to find the highest probability types for a certain person.
def get_top_types(name, inference):
    marginals = inference.get_marginals('{}_type'.format(name))[0].data
    return sorted([(i2t[i], p) 
                   for i, p in enumerate(marginals)], 
                  key=lambda x: -x[1])[:10]
get_top_types('Owa', infer)
> [('SiTe', 0.07492),
   ('TeSi', 0.07291),
   ('NeTi', 0.06628),
   ('NiTe', 0.06628),
   ('SeTi', 0.06628),
   ('TiSe', 0.06628),
   ('TiNe', 0.06628),
   ('TeNi', 0.06628),
   ('SiFe', 0.06075),
   ('FeSi', 0.06075)] 

get_top_types('Itu', infer)
> [('NeTi', 0.13804),
   ('TiNe', 0.11043),
   ('SeFi', 0.05372),
   ('FiSe', 0.05372),
   ('SeTi', 0.05371)
    ... ]
```
The relations can be found similarly:

```python
def get_top_rels(name1, name2, inference):
    marginals = inference.get_marginals('{}_{}_relation'.format(name1, name2))[0].data
    return sorted([(i2r[i], p) for i, p in enumerate(marginals)], key=lambda x: -x[1])[:10]
get_top_rels('Itu', 'Urb', infer)
> [('mirror', 0.06969),
   ('identity', 0.06966),
   ('congenerity', 0.06586),
   ('supervisee', 0.06586),
    ... ]
```

## Other specifications
A few helper functions like `prob_type`, `prob_has`, and `prob_relation`
had to be created to specify the network. In a way we're 
building a relational modeling language.

It would be interesting to see how difficult/tedious other
 [relational learning](https://en.wikipedia.org/wiki/Statistical_relational_learning)
frameworks like 
[Markov logic networks](https://en.wikipedia.org/wiki/Markov_logic_network),
[Probabilistic soft logic](http://psl.umiacs.umd.edu/),
 and 
[Relational Markov networks](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.146.4933) 
are for the same problem.

## Empirical profiles
Social networking sites 
(and [dating sites](http://www.okcupid.com/tests/the-socionics-test-1)) 
have tons of interaction data that 
can be used to validate models like these. 
Maybe the most likely latent-variable model with four dimensions
corresponds roughly to one of the existing models.
Or maybe they're not even close.
Or maybe there isn't a (smallish) point of diminishing
returns when modeling human interactions. It's going
to be interesting to see how these models evolve.

