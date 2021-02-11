---
layout: post
title: A taste of a taste of rewrite systems
---
[Rewriting](https://en.wikipedia.org/wiki/Rewriting) is the process of 
repeatedly replacing terms in a formula with other terms. 
For example, the rule $$\neg\neg x \rightarrow x$$ can be applied to the formula 
$$\neg \neg \neg \neg a$$ to get $$a$$.
This seems important but I needed a bit more background to understand it. 
<!--especially as an engineer trying to learn some good old fashioned AI.-->
These are my notes on some of the concepts in the survey paper 
[_A Taste of Rewrite Systems_](https://link.springer.com/chapter/10.1007/3-540-56883-2_11).

## Example
Rewriting is quite general, so let's start with the Grecian Urn example.

*An urn contains 150 black and 75 white beans. You remove two beans. 
If they are the same colour, add a black bean to the urn. 
If they are different colours, put the white one back.*

If you repeat the process of removing two and putting one back, 
what is the colour of the last bean? Or does it depend?

The system is expressed as the following four rules:

$$
 black\ black \rightarrow black \\\\
 white\ white \rightarrow black \\\\
 white\ black \rightarrow white \\\\
 black\ white \rightarrow white
$$

A possible sequence of (a smaller urn of) beans might therefore be:

$$
black\ white\ black\ (black\ white) \\\\
black\ white\ (black\ white) \\\\
black\ (white\ white) \\\\
(black\ black) \\\\
black 
$$

where the last two beans on each line, in brackets, are substituted according to the rules.

The study of rewriting tries to answer questions like this:
For which types of systems are the answer predetermined, and which systems
terminates? 
Actually doing the rewriting is just computation, and can in fact be used 
to interpret other programming languages. 

## Formulae and terms
A well-formed formula is a sequence of symbols in a formal language, for example
$$ black\ white\ black $$
in the urn example, or $$\neg \neg \neg \neg a$$ in propositional logic.

A formula is composed of terms. 
A term is either a variable, a constant, or a term built from the 
function symbol and other terms, for example $$f(g(a,b),c)$$, 
where $$a$$, $$b$$, and $$c$$ are constants,
$$g$$ is a binary (takes two terms) function symbol, and $$f$$ also. 
Usually I would think of $$f$$ or $$g$$ as taking two arguments, but it seems 
the word "argument" is carefully avoided in the wikipedia article, so maybe 
they only become "arguments" when assigned semantics.
Prolog talks about terms as function arguments though.

## Termination
A system is *terminating* if there are no sequences of derivations that are infinite.
For example, if there are two rules that just rearrange things then it is possible 
to use those rules for ever.

The urn example, on the other hand, is terminating since the number of beans decrease 
every round.

A *termination function* can be used to characterise terminating systems.
The paper lists seven ways in which a function can be a termination function.
I repeat them here each with an example.
- A function that returns the outermost function symbol of a term, with symbols
 ordered by some precedence. For example for $$f(g(a,b),c)$$, return $$f$$, where 
 $$g\succ f\succ c\succ b\succ a$$ could be a precedence (although the ordering only has to be partial).
- A function that extracts an immediate subterm at a position that can depend on 
 the function symbol. For example, a function that always extracts the second 
 subterm for $$f$$. For $$f(g(a,b),c)$$ it would extract $$c$$.
- A function that extracts an immediate subterm of some rank, where the path rank is 
 defined recursively below. For example, if we extract the largest subterm, then 
 for $$f(g(a,b),c)$$ the options are $$g(a,b)$$ or $$c$$, and the largest one would depend 
 on the path rank.
- A homomorphism from terms to some well-founded set of values. 
 A homomorphism is a map (basically a function) between two sets so that operations 
  (stuff like addition and multiplication but the algebraic abstractions) are preserved.
  For example, in our case, a homomorphism from terms to the natural numbers would be 
  the size (number of symbols), and depth (how nested the terms are).
  A well-founded set is the normal set in ZFC, a set that does not contain infinitely 
  nested sets. For example $$\{\{a\}, \{b\}\}$$ is well founded, but $$\{\{\dots, a\}, \{b\}\}$$ 
  is not if the dots represent an infinitely deep recursion. The natural numbers are well-founded.
- A monotonic homomorphism with the strict subterm property, from terms to some well-founded set.
 We already know what a homomorphism (structure preserving map) is and what a well-founded set is 
 (e.g. natural numbers). A homomorphism is monotonic with respect to the ordering if the value 
 it assigns to a term $$f(\dots s \dots)$$ is greater than the value it assigns to $$f(\dots t \dots)$$ 
 if the value of $$s$$ is greater than $$t$$. 
 For example, size has this property: $$f(g(a,b),c)$$ has 4 symbols, and $$f(g(h(a,b)),c)$$ has 5, so 
 the latter is larger than the former. 
 But the first subterm of each $$g(a,b)$$ has 3 and $$g(h(a,b))$$ has 4, so whenever these subterms are 
 larger, the terms are larger also.
 This property is *strict* if the value is strictly larger ($$\succ$$ and not just $$\succeq$$).
- A strictly monotonic homomorphism with the strict subterm property. 
 As above, but the value assigned to the outer term is strictly larger when the inner terms are larger.
 The previous function allowed outer terms to be equal or larger if the inner terms were larger.
- A constant function.

These conditions define a whole set of possible functions that can be used as termination functions.

Now we use these functions to define a *path ordering*:
Let $$\tau_0 \dots \tau_k$$ each be a (possibly different) termination function.
Let the first $$i-2$$ be strict monotonic homomorphisms specifically, and the $$i-1$$th function 
either strict or not strict. 
The path ordering defined by the precedence relation $$s\succeq t$$, where
$$s=f(s_1,\dots,s_m)$$ and $$t=g(t_1,\dots,t_m)$$ are terms and their subterms,
if either of the following conditions hold:
1. $$s_i \succeq t$$ for some $$s_i$$, or
2. $$s \succ t_1,\dots,t_n$$ and $$\langle \tau_1(s),\dots,\tau_k(s) \rangle$$ is lexicographically 
 greater or equal to $$\langle \tau_1(t),\dots,\tau_k(t) \rangle$$, in other words if
 the first elements are greater, or if greater or equal, the second elements and so forth.
 
Dershowitz remarks that this definition combines syntactic (the first three types of termination functions)
and semantic considerations (the others).
 
 Let me try to rewrite this definition in my own words:
 We want to be able to say which terms comes before which other terms, given 
 a bunch of termination functions that we choose.
The induced path order defines a precedence by comparing any two terms $$s$$ and $$t$$.
A term succeeds another term in two cases. 
Firstly, if at least one of its subterms succeeds the function symbol of the other term.
Secondly, if its function symbol succeeds all the subterms of the other term and 
the first few termination functions of the term succeeds that of the other term.

For example, let us compare $$s=f(g(a,b),c)$$ and $$t=f(g(a,c),c)$$ using two 
termination functions, $$\tau_1$$ is the number of symbols homomorphism,
and $$\tau_2$$ a function that returns the second subterm of $$g$$.

We now recursively evaluate the conditions:
1. Does a subterm of $$s$$ succeed $$t$$?
    - Does $$g(a,b)$$ succeed $$f(g(a,c),c)$$?
        1. Does a subterm of $$g(a,b)$$ succeed $$f(g(a,c,),c)$$?
            - Does $$a$$ succeed $$f(g(a,c,),c)$$?
                1. No subterms
                2. Does $$a$$ succeed all subterms $$g(a,c)$$ and $$c$$? No -- $$a\nsucc c$$.
            - Does $$b$$ succeed $$f(g(a,c,),c)$$?
                1. No subterms
                2. Does $$c$$ succeed all subterms $$g(a,c)$$ and $$c$$? No -- $$c\nsucc c$$.
        2. Does $$g(a,b)$$ succeed $$g(a,c)$$ and $$c$$ and is $$\lanble \tau_1(s), tau_2(s) \rangle \succeq \lanble \tau_1(t), tau_2(t) \rangle $$??
    - Does $$c$$ succeed $$f(g(a,c),c)$$?
2. Does $$s$$ succeed subterms of $$t$$ and is $$\lanble \tau_1(s), tau_2(s) \rangle \succeq \lanble \tau_1(t), tau_2(t) \rangle $$?


## Conclusion

For now, I don't have time or space to go into the rest of the paper that 
introduces
Confluence,
Completion,
Validity,
Satisfiability,
Associativity and Commutativity,
Conditional Rewriting,
Programming, and
Theorem Proving.
