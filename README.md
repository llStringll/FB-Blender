# FB-Blender
Unofficial implementation of Facebook's Blender chatbot 

Uses poly encoder implementation as in - [https://github.com/llStringll/Poly-encoders] and transformer based seq2seq model as implemented in HuggingFace, both pretrained on reddit pushshift.io and convAI2, and then fine-tuning end to end keeping Poly encoder frozen on BST.

Training regime:
- A Poly-encoder was trained on reddit pushshift.io using pre-trained BeRT from HuggingFace and same pre-trained tokenizer. Then, it was fine tuned on convAI2 using train_self_original.txt, train_other_original.txt, model being the user with persona.
- A seq2seq was pre-trained from scratch on reddit pushshift.io with tokenizer also trained on same dataset. Then, this was also fine tuned on convAI2, concatenating persona and both speaker's conversation history, each with a different segment id and a seperator token in between. Unlikelihood loss was also optimised for this, with negative labels being labels of other entries in batch, oppoosed to what was done in the paper, where they were calculating the negative tokens by comparing frequency of generated n-grams with that of human uttered n-grams.
- Both models were connected with Poly encoder at higher end to pick most semantically similar sequence, and this picked sequence is concatenated with convo history and fed to seq2seq model and this setup is fine tuned end to end on Blended Skill Task, with Poly Encoder frozen.
- During end to end training, the sequence to be concatenated(i.e., picked by poly-encoder), is replace alpha percent of the times with the actual gold label, because as mentioned in the paper, and is quite obvious, that if the picked sequence was simple concatenated 100% of the times, the seq2seq model will just ignore it, as it was pretrained on reddit and convAI2 for similar task, it will not leverage information from that part of the input sequence, but if it is replace by the actual gold label, it recognizes that part of the sequence being exactly similar to the required label, hence it is forced to pick information from that part.

Predictive response length classifier for seq2seq part to put a hard constraint on generation of <eos> token, hasn't been added yet!

*Training and eval scripts are yet to be added*
