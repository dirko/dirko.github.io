---
layout: post
title: Haskell N-gram text generator
---

After the excitement of the idea of 
[character N-gram language models](),
I played around with a simple N-gram counter and text generator in Haskell.

## Character level language models

I like the idea of character level language models (or even byte level)
because that is the level
on which all our text is serialized. People are creative with text and 
things like ascii-art, novel emoticons, or neologisms and portmanteaus
are impossible to handle with word-level systems.

Ultimately I want to learn a maximum entropy model in Haskell, but I'm still
learning the language. So here is what I have so far. 

## Training and generating script

Let's import some packages first.

```haskell
import qualified Data.Map as M
import qualified Data.List as L
import qualified Data.Set as S
import qualified System.Environment as E
import qualified System.Random as R
```

There are two modes to the script. The `count` mode takes as input a stream of
text and outputs N-gram counts. The `apply` mode takes N-gram counts as input
and outputs a generated stream of text.

```haskell
main = do
    args <- E.getArgs
    gen <- R.getStdGen
    case args of "count":n:_                -> interact $ show . getCounts (read n)
                 "apply":len:temp:start:_   -> interact $ apply gen (read len) (read temp) start . read 
                 _ -> do putStrLn "Usage: ngram count|apply" 
                         return ()
```

## Counting N-grams

First we'll need a place to store the counts. For this we use a 
map where the key is a short piece of text and the value the number of times
that text appears in our training data.

```haskell
type Ngram = M.Map String Int
```
For example this will store 3-grams like

```haskell
fromList [("\n\n\n",2),
          ("\n\nA",370),
          ("\n\nB",483), 
          ...
          ("zze",1),
          ("zzl",4)]).
```

Now we get the set of characters (`vocabulary`), and the counts by 
grouping and counting N-grams. `n` is the N-gram order.

```haskell
getCounts :: Int -> String -> (Int, String, Ngram)
getCounts n x = (n, vocabulary x, count n x)

count :: Int -> String -> Ngram 
count n x = M.fromList . map (\ax@(x:xs) -> (x, length ax)) 
    . L.group . L.sort . take (length x - (n - 1)) 
    . map (take n) . L.tails $ x

vocabulary :: String -> String
vocabulary = S.toList . S.fromList
```

## Sample from a Markov chain

Initially I thought we'd be in trouble here because we'd need a `Random` monad
and I haven't figured out how to sample from it a variable number of times. But
it turned out to be very simple to fold over `R.randoms`.

```haskell
apply :: R.StdGen -> Int -> Float 
    -> String -> (Int, String, Ngram) -> String
apply gen len temp x (n, vocab, m) = 
    foldl (\acc rand -> step n vocab m temp acc rand) 
    x (take len $ R.randoms gen)

step :: Int -> String -> Ngram 
    -> Float -> String -> Float -> String
step n vocab m temp x rand =
    x ++ (sample rand vocab 
    $ map (candidateCounts n m temp x) vocab)

candidateCounts :: Int -> Ngram -> Float 
    -> String -> Char -> Float
candidateCounts n m temp x v =
    temp + (fromIntegral 
    $ lookupDefault 0 (constructNgram n v x) m)

constructNgram :: Int -> Char -> String -> String
constructNgram n v x = (drop (length x - n + 1) x) ++ [v]

lookupDefault :: (Ord b) => a -> b -> M.Map b a -> a
lookupDefault def key m = case M.lookup key m of Just x -> x
                                                 Nothing -> def

sample :: Float -> String -> [Float] -> String
sample r vocab p =
    [vocab !! (last $ L.findIndices (>=r) (accumulate p))]

accumulate :: [Float] -> [Float]
accumulate p = map normalize (accumulated p)
    where accumulated p =
        L.init $ foldr (\x acc -> (head acc + x):acc) [0] p
        normalize x = (x) / (head $ accumulated p)
```

## Example
Now let's generate some Shakespear!

```bash
$ cat input.txt | ./ngram count 2
(2,"\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcde
fghijklmnopqrstuvwxyz",fromList [("\n\n",7223),(
...
("zw",3),("zy",5),("zz",11)])
```

Let's pipe the N-grams to `apply`:

```bash
$ cat input.txt | ./ngram count 2 | ./ngram apply 100 1 'The'
Therd lind's!
N spepaly w wa shevedid; Wh! y cal hald ad s ppases.
chan the f,
```

## N vs smoothness

When we increase the N-gram order, the quality of the output increases. 
Up until a point where the training data is too sparse and the default count of
1 for unseen N-grams is too high. (Smoothing will help here but I'll leave that
for another day).

```bash
$ cat 5grams.txt | ./ngram apply 100 1 'The'
ThehLJmFPpGeiPUGV$KSkjm!R:lVm

fbX-jFEtwPb.GHiq.:jxLmCqQmm?;TyGMHXkFqvkjqvvheeci3ho3$ErNtKaCiSHVCSOo&ob
```

The weight of unknown N-grams steers the Markov chain 
into uncharted territory. We have to lower the 'prior count' of unseen
N-grams, but even so it sometimes snaps into an even gibberisher mode:

```bash
$ cat 5grams.txt | ./ngram apply 100 0.001 'The'
There we hers,
Shall weak of gripe,
And withou,
Though;
Turns.

KINGHAM:
Say, than that the betters.
Hark! thee, good grey-eyed him
Doth not so;
In sicklingbroke harm.

LORD FITZWATER:
Why, and is is throw our harlot sparel justice a fi&wStPFSnhTOmN$zWp$KNyTXjz&vG&HWgzyIc E3S;PJ3FC&QSV!pSKxlje-SoZgDKMtRdZsTEevzJ!-PHhoJS?WV?MsY$RV
ZzmCusA'd,$
RvQWC3 QPvl
sLa&yjoR&DCzspk-
 Kiss you give:
 Your good with us, sea-side.

 BIONDELLO:
 Those leasurest?
 ```




