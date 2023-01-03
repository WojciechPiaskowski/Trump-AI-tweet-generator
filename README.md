# Trump AI tweet generator

# Overview
Project that uses NLP deep learning methods to create an AI model that generates tweets (text) based on Donald Trump's tweets dataset from Kaggle.

# Data Sources
Donald Trump tweets: https://www.kaggle.com/datasets/austinreese/trump-tweets  
Harry Potter books: https://www.kaggle.com/datasets/balabaskar/harry-potter-books-corpora-part-1-7

# Data Issues
Trump tweets dataset contains around 43 000 of tweets, which is about 818 000 words.
Harry Potter books dataset contains around 97 000 of sentences, which is about 1 117 000 words.

Both of these datasets are by far not large enough and also might have a small vocabulary size (45k and 65k) for an effective text generator.
Twitter dataset also suffers from a lot of irrelevant characters (links, emails, embedded images etc.) - most of which were removed in the pre-processing step, however not all of it.
Perhaps it was a foolish task because of the issues outlined above, however the project taught me a lot, thus was worth the time and effort.

# Initial approaches

# GAN
My first approach was to use a Generative Adversarial Network (GAN) in its basic form to generate text, similary as it generates images utilizing the generator and discriminator architecture.
Main issue with that model is that GANs require the data to have a continuous representation, which is difficult for text, since words are naturally discreet.
I did experiment with word2vec and 300-dimensional representations of the data corpus, however with very poor results, such as the example below:

> "Honestly does THE_PRESIDENT_Yes %_#F########_3v.jsn badgered is suppress_dissent Aggressive 've my Duplicating I have Republican Moreover the want"

> "Is second QUESTION_Okay I the that be should which only Republicans the He BY_STEVE_KORTE he the was 2Q_Profit_Falls leader_VK_Malhotra the we the want but the great CHARTS"

Tweets were generated as a sequence of embeddings that were later decoded by Word2Vec with the .most_similary() function.

# LSTM
The second approach was to use a Long Short-term Memory neural network.
It makes more sense than GAN. The model should find a representation of a sequence of characters (embedded as simple integer tokens) or words (word2vec embeddings) and based on it return the next probable character/word.

The problem with character-level prediction is that the model very quickly converged on sequences of the same words that are simply most likely based on the dataset.
Since it only took 70 previous characters (a few to 20 words) and had very simple embedding, it was difficult for the model not to overfit, even with a lot of dropout neurons.
Larger lag than 70 was not possible computationally on my local machine.

**Example output:**
> "that the democrats are all the best the president of the world is a great security and the world is"

> "the best the border and the world than the senate that is a great security and the world than a"

Output above is "deterministic" in the sense that model always takes the characters with the highest predicted probability.

**Example stochastic output:**
> "on this country this morning and scstidkn momet. bih hervo! its gata uo scam, send, if we run why"

> "the west afsertatimns what nerer know that shansoo garoers oe fars, paeio and cons qajd and left ba"

> "approval right with many years and kanalee vhich, and heapt navioniise to people aseing and not"

Stochastic in a sense, in that the models picks the next character based on predicted probability distribution.
This leads to more variation and "unstucks" model from the loop of most common sequences of a few words, but in the end it mostly generates gibberish.

Word-wise text generation, on the other hand, was impossible on my local machine with enough lag due high-dimensional representations of longer sequences would take too much memory.

# Transformer Decoder
Final and the best approach was to use a transformer-based decoder network architecture.
Final implementation uses a tokenizer that later passes the data to a token and position embedding layer (256-dimensional) and further through a few decoder layers.
The whole network is composed of 35M parameters and ends with a vocabulary-sized (around 65k) softmax activation layer.

# Results
Inference-wise the output is not perfect, the issues with data itself outlined in the begining of readme still persist. 
One thing to consider is whether to generate text with the most probable next word or based on the probability distribution of top N words. Below are some examples of the output:

**First 5-6 words are user input.**

**Only the highest probability:**
> "The focus should be on the united states and our country is not a very good thing. "

> "America is the only country that we need to make america great again!  " 

> "Europe will need to think the democrats to get the border and the u.s. will be a very strong and very strong and border security and our military, vets, and will be the great"

> **"Voters showed the importance of the united states supreme court justice reform is a very good thing.    "**  

**Top 2 words, based on probability distribution:**
> "The focus should be on our great american worker who is a total loser!    go  to     a champion   trump   corrects them!     trump thanks.    we are"

> "America is the only country needs to make america great again for the great again!   trump will be back on  nbc. thanks for the best of   the   best ever! "

> "Europe will need to think the u.s. is a total waste of money to our country and then we need a very bad deal. they will be the right thing to get the same"

> "Voters showed the importance of the united nations governors. its a big mistake by the failing new york times for the united states, that they were very bad for our country, not even more"

**Top 3 words, based on probability distribution:**
> **"The focus should be on your healthcare on immigration laws now! we will continue to make our country safe again! maga kag  maga  kag    maga kag     thank you!  maga   maga agenda."**

> "America is the only country in our country and we are not going on a great job.  thank you for your a.  edison   to  see  the  best wishes.   the best of the art of the"

> "Europe will need to think big tax cuts and we will be a big tax cuts. we have the right now. great state is the best in u.s.   history. the american people"

> "Voters showed the importance of the democrats are doing very well as they have been going to get rid of. no collusion, so many are the democrats to take care of the dems and"

**Top 5 words, based on probability distribution:**
> "The focus should be on a lot of good start. they are doing great!      maga   tickets:  ciuzfcjei  gop cont  gop bei trump tower   !   thanks for usa!  i love"

> "America is the only country needs you for all time. we must stop this country! makeamericagreatagain  trump trumparmy jealousy to make america safe!  maga    agenda.  i am a great job.  its time you got it to make"

> "Europe will need to think that the american workers and their oil. people who want them to go on and others have gone on and now the dems are not to keep our government"

> "Voters showed the importance of all time that i will never let this deal to protect our country.  we will make america great!  we must stop the u.s. embassy and secbernhardt! military"

As you can see, the model does output sensible sentences at one time, and complete gibberish at another. That is a problem with data quality, quantity, and computation limitations.

# Harry Potter AI Generator
I did run the same model trained on Harry Potter books instead, with similar, though maybe a bit better results:

**First 5-6 words are user input.**

**Only the highest probability:**
> **"Harry came to the realisation that he was going to be able to see the other side of the table "**

> "Ron and Harry became aware that he was going to be able to see what he was going to be able to get into the castle    "

> **"harry said loudly that he is not going to be able to get out of the room "**

> "harry thought of him as he had been thinking about the same time  "

**Top 2 words, based on probability distribution:**
> "Harry came to the realisation that had been a bit of a new books   and then he had been stuck to the ground   and he had been able to see how to be in a great hall for the time"

> **"Ron and Harry became aware that hermione was sitting on the floor with a small smile**    on the floor and pulled on the ground and looked around at the floor      and then a few feet from"

> "harry said loudly that he is not to see the other people who had been able to see the others in the kitchen  with the rest  of it and was a few minutes"

> "harry thought of him as he had been thinking hard     to see what he had been in his own life        in the same"

**Top 3 words, based on probability distribution:**
> "Harry came to the realisation that had not heard of his voice from under his breath  as he had seen the snitch  and had been a short pause while the latter had to a little   to the other than ever since"

> "Ron and Harry became aware as **they were looking up  and down at each other, and then said, “i think you can tell me what you know about the wizarding world, and the ministry of magic, or else** would do"

> "harry said loudly that **he is not going to have a great deal of evil and that he has never seen**  it in a great deal of time in a great deal of do-it-yourself"

> "harry thought of him as the first years had never heard a new spring to look of the same  time, and his scar on his forehead     was not to"

**Top 5 words, based on probability distribution:**
> "Harry came to the realisation that had not forgotten to tell him  what was happening at the same   time in the forest  floor and had been caught the way he had been all-powerful to look up in his mind to the"

> "Ron and Harry became aware about that they were not discussing what he might have been a chance for a moment, when harry saw a few seconds he had done so much    as much as they heard a jet black had"

> "harry said loudly that he is very much more to see how he would not see whether it would have known it would be in his life, it was so badly if he was sure"

> "harry thought of him as the first years ago  — he saw it — ” but the one was standing there with a flash of fire in a great deal with a loud"

# Conclusion
Model Trained on Harry Potter books is clearly an improvement over the model trained Trump's tweets. The improvement comes from a few factors:
- data quantity - larger corpus,
- "cleaner" data - less pre-processing applied,
- larger and more accurate vocabulary used.

Further research / continuation of this project could start with:
- gathering more data,
- extending pre-processing actions to prepare the dataset better before training the model,
- extending the model architecture to a larger and less computationally restricted.

Alternatively, transfer learning could be utilized in order to leverage existing Large Language Models (LLMs) such as GPT-3 or Blossom.





