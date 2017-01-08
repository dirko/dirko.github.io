---
layout: post
title: Vim is an interface specification
---

In my opinion there are two Vims: the interface (modal editing),
and the environment (plugins and customizations).

## Vim is an interface spec 
With programming languages you should similarly differentiate between the language specification and the tools
to execute or compile that language. In Haskell this is clear - `haskell` is the name of the language while `ghc` 
is the name of the most popular compiler. In Python, on the other hand, people often confuse the specification
and the reference implementation `cPython`.

The most useful part of Vim to me is the interface - I'm happy to move the environment somewhere else because I'm not too
invested in Vim-specific plugins. For example I already use IntelliJ's excellent IdeaVim plugin, and the Vim mode 
provided by [codemirror](https://codemirror.net/) in Jupyter notebooks.

## Org-mode 
I've always been curious about org-mode. Up until now I kept all my random ideas
and todo lists open in a markdown file. I opened the file in Vim with a hotkey <kbd>space</kbd> + <kbd>⌥</kbd> 
that can be configured in iterm, together with a visor-like dropdown.

This year, however, I've finally made the resolution to try out the real thing.

## Spacemacs
[Spacemacs](spacemacs.org) is an emacs distribution with nice defaults and a plan for lasting peace 
and the end to the editor wars.
It has allowed me to become hooked on org-mode without having to take weeks to build a custom 
initialization file, and more importantly without having to give up my Vim muscle-memory. I'm also very
impressed by the default plugins like `helm` and `shell`.

For now, while I still have a life outside emacs, I use [Snap](http://indragie.com/snap/) to quickly switch from whatever
I'm doing to emacs and back.
