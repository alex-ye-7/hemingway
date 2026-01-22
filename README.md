# HemingwayBot

## Alexander Ye

Code based on Andrej Karpathy Neural Networks Zero to Hero Playlist on Youtube

Created a language model that models Ernest Hemingway's writing. 

## bigram.py

A bigram language model assumes that the probability of a word (in this case, one character) depends only on the word immediately preceding it.

For example, "I love cats" has bigrams (I, love) and (love, cats).

Within 4500 iterations, the simple model is able to achieve a cross-entropy train loss of 2.3063 and validation loss 2.2646. This will be our baseline to improve upon.

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


## bigram_v2.py

The bigram language model version 2 applies the concept of self-attention to improve the context-awareness of the model. 

This involves adding positional encoding on top of existing token identiy embeddings. We then implement a head of self-attention, which utilizes the key, query, and value vectors. 

#### Single head of self attention 
Within 4500 iterations, adding a single-head of self-attention of head size 32 results in a cross entropy train loss of 2.2216 and valiation loss of 2.2162. The text is still indecipherable though. 

#### Multi headed self attention 
Within 4500 iterations, adding multi-headed self-attention with 4 heads size 8 results in a cross entropy train loss of 2.0930 and valiation loss of 2.0904. The text is beginning to resemble English.

#### Multi headed self attention with a feed forward
Within 4500 iterations, adding multi-headed self-attention with 4 heads size 8 results in a cross entropy train loss of 2.0813 valiation loss of 2.1042. The text is beginning to resemble English and in dialouge format.

#### Transformer block 
Within 4500 iterations, adding multiple blocks of attention + feed forward, along with residual connections results in a cross entropy train loss of 1.8732 valiation loss of 1.9530. The text looks like English with some identifiable words, but still nothing coherent.

The final version adds in dropout and layer normalization before the transformations. 