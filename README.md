# character_generator

A character-by-character text generator using an recurrent neural network (RNN).

Note: the code has been updated to use tensorflow's higher-level APIs. This README needs to be updated.

# Fun Examples

With the [Bible](http://www.bibleprotector.com/TEXT-PCE-127-TAB.txt):

After 1 training epoch:
> Badar amr therean; tre peaslt Masibting in from receta that is life in the tonars.
> And hin offlabua there [ari chy] cateons.)
> And Jusail; the king of Jebrishes, and thou [brine] wayes, and bpongyed [halb] to me, but and he syeplign strent to hor; the tomy evin counteth the wind of the altwide befole the tharl not to the flopher, the bolimenss, hath foriesh his fight [is] you said, for he man he spait unto the LORDghting kank the nhirst that thee is a clutt thine trils, and keoHt; are the ralpry.

After 99 training epochs:
> What will not ye suffer, but that [is] the presence of Jesus, which were born unto her for our dust shall dwell therein.
> But we are warned in the sight of God; turn away his ropend: but the wicked shall come, which came to Samaria, and came near serpent, which said, Surely the people now should eat not in the sight of this goodness.
> Yea, testation a soul shall surely be burniturl, which are most nothing to their gods, and lightness upon the circurce unto them.
> For the morning appear to the tongue


# How to Run

It's easiest to use [conda](https://conda.io/docs/user-guide/install/index.html#regular-installation) to install tensorflow and numpy.

The character_rnn can be installed with

```pip install .```

Then run the character RNN on some text:

```./character_rnn/text_generator.py text.txt  > sampled_text.txt```

# TODO

* Rewrite with tests. This is really just a prototype to get something to work. I would like to start from scratch and incorporate a testing framework.
* Use tensorboard to monitor training and validation loss.
* Create a pre-processing step to learn character embeddings.
* Allow the trained model (and future embeddings) to be saved and restored.
* Allow training from multiple sources. For example, multiple authors could be used for training all at once. It could then generate text from any particular one of them. This would save on overall training time and allow the model to learn even from relatively small datasets for a particular author.
* See if it works when using a convnet on character embeddings.
