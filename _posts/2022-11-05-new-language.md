---
layout: post
title: A new language
---

Over the weekend I created a new programming language
with functional programming elements, objects, and macros.

## Language

The language is expression based. 

### Expressions and functions
Let's look at some code. To define a function,
use the `def` function.
```lisp
def(my_plus (a b)
  {a + b}
)
```

To call it, add the arguments in brackets.
```lisp
my_plus(3 2)
>>> 5
```

Here is a function that operate on lists.
```lisp
def(my_function (c)
  {c[0] + c[1]}
)
my_function(list(1 2 3))
>>> 3
```

### Loops
Loops are done with the `loop` function:
{% raw %}
```lisp
defun(fizzbuzz ()
  loop(for i from 0 to 100 do 
    switch(
      {{{i mod 3} = 0} and {{i mod 5} = 0}}
        print("FizzBuzz")
      {{i mod 3} = 0} 
        print("Fizz")
      {{i mod 5} = 0} 
        print("Buzz")
      otherwise print(i)
    )
  )
)

fizzbuzz()
```
{% endraw %}

Note that white-space is mostly not significant.

### Hash-maps
A dictionary (hash-table) can be created with the `dict` keyword.
```lisp
def(run ()
  let((h)
    {h := dict()}
    {h[3] := 10}
    print(h[3])
  )
)
run()
>>> 10
```

Note that we introduce a new variable with the `let` keyword.

### Macros
Macros are defined with `@def`:
```lisp
@def(my_log (f) 
  print(f)
)
log(print(1))
>>> (print 1)
>>> 1
```

### Classes and objects 
Classes are defined with `class`:
```lisp
class(point 
  (x y z)
)
```

An instance of this class is created with `new`:
```lisp
{v := new('point)}
{v['x] := 1}
v['x]
>>> 1
```

Methods with `method`:
```lisp
method(reset_point ((obj point))
    {obj[x] := 0}
)
reset_point(v)
```
Which means that the method specializes `obj` of type `point`.

## Implementation
So this is of course not really a new language. 
You might have noticed a striking similarity to ... common lisp!

Basically I've gone through the noop lisp rite-of-passage by playing with 
custom lisp syntax. 
I can recommend this method to quickly learn more about lisp!

The first change is to implement function call syntax.
There are multiple packages for this -- I used neoteric-expressions from 
the `readable` project. 
```lisp
(ql:quickload "readable")
(readable:enable-neoteric)
```
This also gives us infix notation within curly braces, e.g. `{x + 1}`
is actually read as `(+ x 1)`.
(`readable` also allows white-space based syntax, but somehow I enjoyed playing 
with that less -- just function-call and infix already makes things much more 
familiar.)

Now define and rename our own "keywords" for function and macro definitions:
```lisp
defmacro(def (&rest p)
  `eval-when(
    :compile-toplevel(:load-toplevel :execute)
    defun(,@p)
  )
)
defmacro(@def (&rest p)
  `defmacro(,@p)
)
```

The `readable ` library also allows us to create a custom 
macro for interpreting square brackets by implementing `$bracket-apply$`.
At run time, inspect the type of object outside the brackets and
call the appropriate access method.
(Various disadvantages to this being run-time, but it works.)

```lisp
def(access (e index)
  cond(((eq type-of(e) 'cons)
        (nth car(index) e))
       ((eq type-of(e) 'hash-table)
        (gethash car(index) e))
       (t (slot-value e car(index)))
  )
)

def((setf access) (new-value e index)
  cond(((eq type-of(e) 'cons)
       (setf nth(car(index) e) new-value))
      ((eq type-of(e) 'hash-table)
       (setf gethash(car(index) e) new-value))
      (t (setf (slot-value e car(index)) new-value))
  )
)

defmacro(readable:$bracket-apply$ (e &rest index)
  `access(,e ',index)
)
(readable:enable-neoteric)
```

I've also experimented with custom infix assignment, 
and shortened class, object creation, and method names:
```lisp
setf(macro-function(':=) macro-function('setf))
setf(fdefinition('dict) #'make-hash-table)
setf(fdefinition('class) #'defclass)
setf(fdefinition('new) #'make-instance)
setf(fdefinition('method) #'defmethod)
```

Lastly, let's create an alternative to `cond` that uses less parentheses:
```lisp
def(group-pairwise (list)
  loop(:for (a b) :on list :by #'cddr :while b 
    :collect list(a b)
  )
)

@def(switch (&rest args)
  let((v)
    {v := group-pairwise(args)}
    cons('cond v)
  )
)
```

## Conclusion
I'm also indenting in a style that is more familiar to someone 
coming from languages with algol-style syntax.

Advantages of what is here:
- The syntax is more familiar to most people coming from most other languages.
- Arguably easier to read.
- Can easily mix both styles in the same file.

But after spending about 10 minutes writing lisp I got used to the syntax.
I quickly started to see why more experienced lispers can't understand 
why anyone would be put off with the syntax -- it quickly fades into the background.

Disadvantages:
- Macros do not directly map to the language -- although I was surprised at how easily it still mapped.
- Biggest disadvantage for me was the editor couldn't send top-level 
 definitions to the repl if it started as `defun(...)` 
 rather than `(defun ...)`. So while developing I mostly kept the outside braces.
- Also the editor (I was using SLIME) indented completely differently than what I wanted in this case, so lots of manual re-indenting afterwards.
- The infix assignment (e.g. `{v := 3}`) looks awkward.
- The normal lisp indenting is nice and compact -- no reason to double the lines of code by closing braces on their own lines.
