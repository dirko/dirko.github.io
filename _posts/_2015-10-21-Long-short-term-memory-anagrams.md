---
layout: post
title: Long short term memory anagrams
---

[Long short term memory (LSTM)]() neural networks are a type of recurrent
neural network that can model sequences. I tried to get them to produce 
anagrams and it sort of worked.

## Long short term memory memory

I was first curious as to how well a LSTM network can remember what text 
you just showed it. To test this let's show it some characters followed by
a separator character followed by the same characters:

```python
example = '''abcbdd|abcbdd
             cdc|cdc
             dd|dd
             aaaaaa|aaaaaa'''
```
