---
layout: post
title: Loopy glitches
---

While developing a loopy belief propagation library I came across a bug that produced
this interesting image. 
<div style="overflow:hidden;">
   <img src="/images/2019-03-06-Loopy-glitches/lbp_small.png" alt="Loopy belief propagation glitch image" style=" clip-path: inset(0px 2.0px);width:45%" />
</div>

The code that generated the glitch can be found [here](https://github.com/dirko/pyugm/commit/10f084e4ad9e03c86b3cac5cd488a2d2bfad7eee) 
(and the fix [here](https://github.com/dirko/pyugm/commit/633d5322287838aaa6fb64f6e8988313788d9420)). I was trying to do 
image segmentation on a frame from Spongebob, but instead of smoothing out the foreground/background
beliefs the beliefs became unstable and flowered into something that looks to me like a figure.

Later I wondered how the beliefs evolved over time, which results in the cellular-automaton-like sequence: 
<img src="/images/2019-03-06-Loopy-glitches/beliefs.gif" alt="Loopy belief propagation glitch beliefs over time" style="width: 45%"/>

I'm still not completely
sure what caused the glitch, but it seems that I previously tried to use `int8`s with `numba.jit` and 
in my case only `int32` worked. Maybe an overflow or types not lining up between Cython, Numba, and Numpy.
