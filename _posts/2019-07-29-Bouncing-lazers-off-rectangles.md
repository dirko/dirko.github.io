---
layout: post
title: Bouncing imaginary lasers around grids
---
<script src="{{ base.baseurl | prepend: site.baseurl }}/static/2019-07-20/bounce.js"></script>
<canvas id="myCanvas" width="500" height="400"></canvas>
<script>
console.log('canvas');
var canvasElement = document.querySelector("#myCanvas");
console.log('cal');
solveR(canvasElement, 25, 20, 20, skip=false);
</script>

Since I was small I would play a little game on tiled floors to pass the time. 

You shoot an imaginary laser beam from the bottom-left tile 
that travels diagonally upwards until it reaches the wall. Then it bounces 
against the wall and continues leftwards and upwards. 

It bounces around the floor like that until it reaches a corner. There it dies. 

<script src="{{ base.baseurl | prepend: site.baseurl }}/static/2019-07-20/animate.js"></script>
<canvas id="anCanvas" width="180" height="240"></canvas>
<canvas id="anCanvass" width="28" height="240"></canvas>
<canvas id="anCanvas2" width="360" height="240"></canvas>
<script>
console.log('canvas');
var canvasElement = document.querySelector("#anCanvas");
console.log('cal');
startBlocks(canvasElement, 3, 4, 60);

var canvasElement = document.querySelector("#anCanvas2");
console.log('cal');
startBlocks(canvasElement, 6, 4, 60);
</script>

The game is to predict in which corner it will end---for every different sized 
floor it would end in one of the three opposite corners from where it started.

To see if there is a pattern, let us play the game on different sized floors.
For a 1x1 floor, the game immediately ends in the "top-right" block (well technically 
there is only one block so it ends in all four, but let's choose the top right most 
because it fits the rest of the pattern).

For any NxN block it will immediately go to the top right most corner. 
So we draw a triangle in the (N, N) block to represent the game ending there.

For NxM sized floors we can do the same, each time drawing a triangle in the 
(N,M) block so that the legs of the triangle meet in the direction of the corner 
where the game ended.

In this way we get the pattern at the start of this post.

Puzzle: *What is the formula to determine the ending corner for any NxM floor?*
