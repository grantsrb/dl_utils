import sys
sys.path.append("../")
import tokenizer
import numpy as np


texts = [
    (
        "The grid is 8x8 units with a 0 deg rotation. The player is at (0,4), there is a Bramble located at (2,5) with a 0 deg rotation, a Fish located at (3,3) with a 180 deg rotation, a Tower located at (1,4) with a 180 deg rotation, a Pointer located at (3,4) with a 270 deg rotation, a Bramble located at (4,6) with a 0 deg rotation, a Bird located at (2,3) with a 0 deg rotation, a Beedle located at (0,6) with a 90 deg rotation, a Magnet located at (1,3) with a 270 deg rotation, the reference object is a Block at (2,4) with a rotation of 180 deg. What action do you take to find a target object?\nHypothesis: targets are Towers. The nearest possible target object to the player is at (0,4) is the Tower located at (1,4) with a 180 deg rotation"
   ),
   (
    "Action: grab\nResponse: incorrect\nHypothesis: targets are Magnets. The nearest possible target object to the player is at (1,4) is the Magnet located at (1,3) with a 270 deg rotation"
   )
]

print("Texts:")
for text in texts:
    print(text)
    print()

tokenizer = tokenizer.Tokenizer()
tokenizer.train(texts)

print("Toks:")
toks, max_len, _ = tokenizer.tokenize(texts, verbose=False, ret_all=True)
for text in toks:
    print(text)
    print()

print("Ids:")
ids = tokenizer(texts).numpy()
for tok,samp in zip(toks,ids):
    print("Sample:", samp)
    tok = [tokenizer.bos_token] + tok + [tokenizer.eos_token]
    targ = np.asarray([tokenizer.word2id[w] for w in tok])
    print("Target:", targ)
    assert np.array_equal(targ, samp[:len(targ)])
    print()

print("Max Toks:", max_len)
print("Max Ids:", ids.shape[-1])
