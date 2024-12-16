---
layout: post
title: Submodularity and the set cover problem
---

If a function on sets is submodular, approximation bounds can be proven for 
optimisation problems involving that function.
One such case is the set cover problem.
This post contains my notes on the problem and show how the approximation bound can 
be proven.

## The set cover problem
*Disclaimer: I don't know what I'm talking about*


The minimum set cover problem is one of the classical NP-complete problems that 
has been extensively studied:
Given a family of sets $$ X = \{\ X_1, X_2, \dots, X_n\ \} $$ and the 
universe of elements $$ U $$ where the universe equals the union of sets 
$$ U = \bigcup X $$,
find the *cover* $$ S \subseteq X $$ with the smallest cardinality so that 
the union of the elements in the cover equals the universe, $$ \bigcup S = U $$.

For example, with $$ X = \{\ \{1,2\}, \{3,4\}, \{1,3,4\} \ \} $$ 
and $$ U = \{\ 1,2,3,4 \ \} $$, the smallest cover has two elements namely
$$ S = \{\ \{1,2\}, \{3,4\} \ \} $$.

## Approximation algorithm
The set cover problem is NP-complete. It is in NP since if we could guess a
cover $$S$$ then we can easily verify whether $$S\subseteq X$$ and $$\bigcup S = U$$.
Proof that it is NP-complete can be done by reducing from 
the vertex cover problem for graphs for example.

So we cannot find an exact polinomial time algorithm for the minimum set cover problem
unless P=NP, which seems unlikely.

So we can approximate.
An obvious algorithm is to iteratively choose the set in $$ X $$ with the largest number of
uncovered elements until all elements in $$U$$ is covered --- the greedy algorithm:

$$ \hat{S} \longleftarrow \{\} $$  
while $$\bigcup \hat{S} \neq U$$:  
$$\quad \hat{S} \longleftarrow \hat{S} \cup \{\ \arg\max_{C\in X} |C\setminus \hat{S}| \ \}$$  

## Approximation ratio
So how good is this approximation?
It can clearly be calculated in polinomial time, but how close is it to the 
exact solution?

One way to quantify the quality of an approximation is with the *approximation ratio*:
An algorithm's approximation ratio is $$\alpha$$ if, for any instance of a problem,
the algorithm gives a solution that is within a factor of $$\alpha$$ of the true optimum. 
In other words, it is a bound on the worst-case performance of the algorithm 
[[1](https://courses.engr.illinois.edu/cs583/sp2018/Notes/intro.pdf)].

One of the ratios that were proven for the greedy set cover is 
$$\alpha=\sum_{k=1}^n \frac{1}{k} \approx \ln k$$, 
where $$n$$ is the number of elements in the universe $$U$$.
(There is also a tighter bound $$\sum_{k=1}^d \frac{1}{d}$$, where $$d$$ is the number of elements
in the largest subset of $$X$$.)

## Submodular functions
So how can we prove that this is the approximation ratio for this algorithm? 

Let's follow, at a high level, the argument in the lecture notes of Deeparnab Chakrabarty 
[[2](https://www.cs.dartmouth.edu/~deepc/LecNotes/Appx/1b.%20Greedy%20Algorithms%20and%20Submodularity.pdf)].

A set function $$f$$ assigns a value to any subset of a universe $$V$$, $$f: 2^V \to \mathbb{N}$$.
A set function is submodular if for any $$A\subseteq B \subseteq V$$, and $$i \in V \setminus B$$,
$$f(A \cup i) - f(A) \geq f(B\cup i) - f(B) $$.

<img src="/static/2023-01-21/sets.svg" alt="Sets" style="width: 60%;"/>

In a way, adding an element $$i$$ to the smaller set $$A$$ increases $$f$$ more than adding that
element to the larger set $$B$$ --- diminishing returns.

## Approximation bound
If we let the universe $$V$$ equal the indices of the sets $$X_1,X_2,\dots,X_n$$, and then
define the set function on a subset $$I\subseteq V$$ as $$f(I) = |\bigcup_{j\in I}X_j| $$, i.e. 
the number of elements covered by $$I$$. 
Then $$f$$ is submodular since adding an element to a smaller set results in an increase of the 
number of elements that were not already in that set, while
adding that element (the index of a set) to a superset of the smaller set might result in a smaller 
increase since more of the elements of that set might already have been in the superset --- so 
the increase is the same size or smaller, but not larger.

The set cover problem asks, find $$I$$ such that $$f(I) = f(V)$$ while minimising the cost $$|I|$$.
I'll leave an overview of the proof to
[[2](https://www.cs.dartmouth.edu/~deepc/LecNotes/Appx/1b.%20Greedy%20Algorithms%20and%20Submodularity.pdf)].

## Conclusion
I still have a couple of questions about the connection, but these notes document a start 
on the types of concepts that is useful when analyzing approximation quality for algorithms.


