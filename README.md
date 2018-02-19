# character_generator

A character-by-character text generator using an recurrent neural network (RNN).

The network architecture starts with an embedding layer. Then a user-defined number of (LSTM + Batch Norm + Dropout) layers. Finally, there's a fully-connected layer + softmax over the labels.

For text generation, there's a temperature option to allow a trade-off between sampling the character probability distribution and choosing the most likely character. I'd like to add beam search for better text generation.

# Fun Examples

These examples come from a network with three LSTM layers with 256 units each, embedding size of 10, and 0.5 dropout. 

The collected works of Jane Austen, 70 epochs of training, temperature of 0.2, and seed text of "The ":
>The property I have a staircase of the acquaintance and see the sister and the say of such a strong and sisters and the sisters and the same time to possibly as!—I am sure I had not even married the same time and silence of being a sort of sense and saying that, in the same time that I have not been a sensible of the same time to be seen the family of the same time to be taken a very serious too much to be aurised to him with. I have nothing to be sure for the attention to their sister’s conduct


The [Bible](http://www.bibleprotector.com/TEXT-PCE-127-TAB.txt) with 70 epochs of training, temperature of 0.2, and seed text of "The ":
>The LORD thy God shall be a son of man, and the strength of the LORD thy God hath seen the LORD that hath seen the word of the LORD thy God.
>After the sons of Benjamin the son of Arawiah, and Abimelech, and Abimelech, she said unto him, The LORD hath sent me before the LORD thy God, and the sons of the sons of Jeroboam the son of Ammon, and the sons of Jeroboam the son of Amminazar, and Abimelech, Jahariah, and Abimelech, and Abihamam, and Amaziah, and Abimelech, and Abimelech, and Taran, and Ga

# How to Run

First install tensorflow, keras, and argh.

Then preprocess the data. It's going to save a numpy array of the characters and a mapping between characters and ids:
```{bash}
./model.py initialize ~/Data/text/jane_austen/concat_jane_austen.txt  -d ~/Data/text/jane_austen/chars/text_data.npy -i ~/Data/text/jane_austen/chars/text_ids.json```
```

Train the model. There are lots of options for the network architecture. If you have a GPU and CUDNN, then use the "--use-cudnn" option. It will be much faster.
```{bash}
./model.py train --epoch 70 -b 128  --use-cudnn -n 3 -r 256 --embedding-size 10 --dropout 0.5 -s 10 ~/Data/text/jane_austen/chars/text_data.npy ~/Data/text/jane_austen/chars/text_ids.json jane_austen_char_0
```

Generate some text!
```{bash}
./model.py generate-text ~/Data/text/jane_austen/chars/text_ids.json jane_austen_char_0 --use-cudnn --temperature 0.2 --embedding-size 10 --rnn-size 256 --num-layers 3 --start-text "The "
```

# TODO

* Have a flag to use words, rather than characters.
* Support variable-length sequences. This may mean giving up the fast CuDNNLSTM.
* Use a beam search for text generation.
* Incorporate with tests and refactor to make the code nicer.
* Have the model dump a history of the mini-batch loss and accuracy.
* Allow training from multiple sources. For example, multiple authors could be used for training all at once. It could then generate text from any particular one of them. This would save on overall training time and allow the model to learn even from relatively small datasets for a particular author.
