# **ALTERNATE REALITY**

### **DESCRIPTION**

---

This is a project that I am working on to apply my newfound skills of Recurrent Neural Networks and Natural Language Processing. This project deals with **text generation** using LSTM RNN's.


### **Motivation**


---

I lived in New York City from 2014 to 2018. In the American education system, students write out different things (essays, poems, research papers, op-eds (opinion editorials), etc.) on a daily basis.

Throughout my time at *The Bronx High School of Science*, I was trained in the art of scientific thinking! I had to write daily homeworks for my english class, most of which I typed up in Google docs. I also wrote numerous research papers for my english and history classes, some of which were well above **20 pages**. All of these texts are my creation. My own piece in the world of words and sentences.

When I had to leave *New York City* in the middle of my senior year, I was devastated beyond narration. It was the single most horrifying experience of my life. I was detached from everything I belonged to and stood for (kind of like Thor if you think about it) and banished away.

Ever since then, I have not been able to think like I used to be able to. I read text, but I am not critically thinking of it, not finding literary and rhetorical devices that the author used to set a scene, or show the development of a character.

And so I turned to **Neural Networks** to help me out. By using **Python**, **Keras**, and **LSTM**, I will be able to create a *Recurrent Neural Network* for language modeling.



### **METHODOLOGY**

---

I used the high level Keras API to create my Recurrent Neural Network. My text data lives in [this](1) Google drive link.

One example of the text I have written is:

> One of the biggest change brought by the American Revolution was the general grudge against the British empire inside of the colonists. It started with the Proclamation Line of 1763 which turned the Ohio River valley into Royal Indian Reserve and forbade colonists from settling west of the Appalachian mountains. The Ohio River valley was the place that the colonists fought the French and Indian war for in the first place. And now, Britain forbids them to live there. This made colonists resent the British empire. When the colonists still settled in the river valley (it was nearly impossible to strictly enforce the proclamation), Ottawa chief Pontiac led a rebellion in western Pennsylvania in May 1763 due to conflicts between colonists and indians in the river valley. Disgusted with Pontiac’s rebellion and the failure of colonial government to keep the people’s interest secured, Paxton Boys of Lancaster killed 6 Indian men, women, and children. The British promised to enforce their agreement with the Indians. After this, Britain passed the Stamp Act in 1765 which imposed a tax on commonly used items like cards, dice, legal documents, newspapers, pamphlets, and advertisements to pay for the debt incurred by Britain during the French and Indian War. Anger and discontent led the colonists to hold the Stamp Act Congress, the Virginia Resolves, boycott British textiles, and violent protests by the Sons of Liberty. The first three were nonviolent protests in hope that the British government would understand their problems because of the Stamp Act. The Stamp Act Congress (first continental congress) even sent a petition for this to Britain. But the Sons of Liberty had different ideas. They tarred and feathered the British officials who tried to enforce the unfair taxes and planted liberty poles in celebration of colonial self government. However, regardless of the way of protest, all colonists strongly resented the colonial government for this direct taxation. However, colonists did not want or were prepared to separate from the British empire yet. But then people read and observed in the 1700s that the British government was not virtuous at all and violated all enlightenment ideas. It didn’t put the public welfare in front of the king’s selfishness, or limited its own power. This was essentially the critical point where the colonists could have been stopped. However, the Boston Massacre of 1770 (British soldiers attacked colonists and killed 5 colonists) and the Boston Tea Party of 1773 (colonists threw tea offboard a ship in protest to tax on it), made the unsure colonists chose their sides. Enraged by such actions, Britain passed the Coercive Acts (Intolerable Acts for the colonists). The Acts shut down all trade ports in Boston which affected trade and commerce in all the colonies, all elected positions such as local judges in Massachusetts were to be appointed by British king, all royal officials received trials in different colony or even in Britain, and increased powers were given to French speaking Catholics in Canadian province of Quebec. As if that was not enough, Britain passed the Quartering Act of 1765, that allowed military commanders to live in houses among the colonists. The colonists also had to pay the rent and give food to the military. These laws were literally openly challenging the colonists to protest and gave a clear message that the British government was not virtuous. In 1774-1775, colonists start talking about breaking apart from the British empire. In 1776, Jefferson wrote the Declaration of Independence which was the definite decision of all the colonies. People could no longer take the oppression of Britain over them. They wanted to drive away from them every British who wanted to see them in “chains of tyranny” to their island of Britain where they would enjoy their slavery and to never let them return to the happy and free land of America (Document A). The colonists have come as far from quietly bearing heavy taxes from Britain to declaring their independence from the British empire. It was without any doubt, a major change brought by the American Revolution.

This is a paragraph from my fall research paper in my junior year AP US History class. I gathered up about 10 of my previously written homeworks, essays, and research papers and merged them into a big text file ([here](2)).

After much research done from previously made similar models by other people, I came to the conclusion that the best way to create (X, Y) pairs with my text data would be to split it into sections of 11 characters, taking the first 10 characters to be the X example and the last character to be the corresponding Y example.

After parsing my data, some variable dimensions looks like:

```python
print(f"Length of corpus: {LENGTH}")
print(f"X.shape = {X.shape}")
print(f"Y.shape = {Y.shape}")
print(f"Number of examples: {m}")

# Length of corpus: 229570
# X.shape = (76520, 10, 96)
# Y.shape = (76520, 96)
# Number of examples: 76520

# m can be changed by changing the step variable in the notebook
```

### **NETWORK ARCHITECTURE**

---

Some hyperparameters that took be a long time to manually tune were

+ Learning rate
+ Batch size
+ Number of LSTM cells
+ Number of LSTM layers

As of now, the best combination I have come up with is

```python
learning_rate = 0.01
batch_size = 8192
n_a = 256 # number of LSTM cells
n_L = 2 # two LSTM layers
```


### **RESULTS**

---

After tuning the hyperparameters, I trained on 100 epochs and managed to score a **93% accuracy on the training set**. This accuracy is decently good considering the amount of data is fed and the computing power I had (sessions of 35GB on Google colab).

These are the first 400 characters my model wrote:

> My name is Ramansh Sharma My name is the public control considered to fut tell overt of a decide the world to life people real and massive really came to congres to be streated it could have ingarding the world many that estabe by nugues and could have been guvern for the Soligition, purched that remamplogr prices repective independence in 1982. Bech of 1980ided to social and enjoyable capting the sent a see that her seperones in Am

Upon examining this creation, it is evident that the model is not at its best. There are multiple spelling errors (ingarding, guvern, etc.) in the text. But on the bright side, the model realizes basic punctuation like commas, period, and capitalization of the first character of the sentence.

I will be playing with the temperature and the earlier mentioned hyperparameters to better tune the model.

### **LIVE CODE**

---

In this repository, I have presented by code as a Jupyter notebook. But I originally coded in Google colab. I am providing the link to the colab for viewing and making it public.

Link - [http://bit.ly/alternatereality](http://bit.ly/alternatereality)

### **CONTACT**

---

If you have any concern or suggestion regarding this project, feel free to email me at [sharmar@bxscience.edu](sharmar@bxscience.edu).




[1]:https://drive.google.com/drive/folders/1AAIv3BWkclfMJnnv8xXt9iQyFns0Cp1R?usp=sharing
[2]: ../master/text_data/merged.txt
