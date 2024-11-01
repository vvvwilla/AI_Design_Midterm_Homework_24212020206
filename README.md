Firstly, we reproduced the Shakespeare training set using Course2's nanoGPT model, using the computer's built-in CPU, 
and the reproduction results were good. The reproduction diagram is shown in the following figure:

![](https://github.com/vvvwilla/AI_Design_Midterm_Homework_24212020206/blob/61aea278e74b50427bac33794ced0f0000a19015/picture/Reproduce%20results.jpg)

Then we optimize the training set and write our own training model, as follows:

Enhance the four modules: model.py, train.py, sample.py, and config/token_char.py.

(1) Introduced the EnhancedTokenizer class, incorporating special tokens (such as , , ) to facilitate more intricate text processing, offering enhanced encoding and decoding functionalities that support the handling of special tokens.

(2) Incorporated relative position encoding, attention temperature parameters, and a gating mechanism into the CausalSelfAttention class to bolster the model's attention mechanism.

(3) Employed a combined activation function of GELU and ReLU in the MLP class, and introduced learnable residual connection weights.

(4) Added a scaling factor for residual connections in the Block class.

(5) Integrated Matplotlib into train.py to chart training and validation losses, enabling better monitoring of the training process, and modified the learning rate scheduler to facilitate a smoother warm-up phase.

(6) In GPTConfig, altered the number of layers, heads, and embedding dimensions, and incorporated dropout to enhance the model's capacity and regularization capabilities.

(7) Included top-k and top-p sampling strategies in the generate method of the GPT class to facilitate more versatile text generation.


Finally, we used our own training set and the test results are as follows:

![](https://github.com/vvvwilla/AI_Design_Midterm_Homework_24212020206/blob/9771947ea0fb4bdd680cd6b1ff393e4484fe0dd8/picture/Training%20Progress.jpg)

The text is displayed as follows：

```
"IN the big of his own when she said her being of yourself.  And you wouldn't like to want to the same.  "

"Yes, you have no manner and when I do to make it might be particular.  "

"I'm sure to say for the world before the room of the boat, I shouldn't do think?"

"And I've got to say it was a brief of the moment.  It's all the moment.”

"I can't be a cottage.  "

"I don't know for the shop of the boat, you must be no many a lot good more with his woman.  
I have not been her sister to be going.  I don't know what we shall be think the sea of home, sir.  
And I shouldn't go to stay the short, and Jessie had to think it was done.  I'm not been to make
up the storm of business.”

"It was to the few that he said.  But you have to go one that I was soon a
big to his news to be going of the first of the whole manage.  I couldn't
be done for the girl's mother."

"And I'm going to say to be a boat.  I'm not like to the drowned!"

"And you are you are your business.  I don't believe her particular lay
even her sort of good with a passe."

"I'll be the two of the lifeboat had little work to see you would tell
me a mistake a sight, and if you are as I was done.”

"AN any sight be sure to her work that they was a little home with the place.  I
shouldn't see he had been remained up her time, I wouldn't mind make me."

"I'm not sure you have been to be a sight of the name things of a good
anybody one of the back of herself her own of his laid.  It's a good face it
had been done.”

It was not given you do to as care it do of you.  Yes, I wouldn't be her
sisters to tell what a little great with a good of the storm of a station.
"I have not got to be comfort to say.  It's a smiled of pause.
He's the matter to be a time more disappose.  I will do not have been to
leave anybody about the boat.  "

"I have got to be to me what it should not go to be longer.  I do think
it was so what I'm sure to her.  And I can give it might be be the way in
a the long wind.”

"Yes, he must be so much as the relief, so that's got more sea.  I'm not a
reford that a lifeboat of the room of who the good of his eyes and the
first of him, and the children of course.  "

"I wish so much him to go a man, not all you may be be so sometimes.  "

"The doctor wouldn't be likely could be a lifeboat was work that she was no
the face.  It's the widow here for a big as a break for the road, and they
want to be many to her find the first.”
```

Our model still has certain shortcomings in data processing and response speed, and we will continue to optimize and enhance our training capabilities.
