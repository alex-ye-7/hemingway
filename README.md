# HemingwayBot

## Alexander Ye

Code based on Andrej Karpathy Neural Networks Zero to Hero Playlist on Youtube

Interested in how large language models work under the hood

Created a language model that models Ernest Hemingway's writing

Without a GPU, how close can I get?

## bigramLM.py

A bigram language model assumes that the probability of a word (in this case, one character) depends only on the word immediately preceding it.

For example, "I love cats" has bigrams (I, love) and (love, cats).

### Implementation Details
**get_batch(split):** Generates a small batch of inputs and outputs for our model based on block_size, or how much context we want to give for next character

So for example, if we have a block of tokens encoded as tensor([67, 64, 28, 16, 13,  1, 21,  9, 26]):
- When context is tensor([67]) target is tensor(64)
- When context is tensor([67, 64]) target is tensor(28)
- When context is tensor([67, 64, 28]) target is tensor(16)
- When context is tensor([67, 64, 28, 16]) target is tensor(13)
- When context is tensor([67, 64, 28, 16, 13]) target is tensor(1)
- When context is tensor([67, 64, 28, 16, 13,  1]) target is tensor(21)
- When context is tensor([67, 64, 28, 16, 13,  1, 21]) target is tensor(9)
- When context is tensor([67, 64, 28, 16, 13,  1, 21,  9]) target is tensor(26)
