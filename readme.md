# Training a small transformer to do arithmetics

A fun project to mess around with tiny transformer model and to better understand how a model do math.

```
Prompt: 19 + 10=
Model Output: 29

Prompt: 5.5 + 1.8 + 0.2=
Model Output: 7.5

Prompt: 16 + 68 + 20.1=
Model Output: 104.1

Prompt: 17 - 10.5=
Model Output: 6.5

Prompt: 16 + 68.9 - 20.7=
Model Output: 64.2
```

## Generating our own dataset
Training data will be strings like "100 + 9 - 3", and the label will be "106". We will generate all our training data programmatically. To keep model small, we will keep length of expression short and only float up to 2 decimals.

Example:
```json
[
    [
        "-51.69 / -59.25 - -98.82", <-- math expression/instruction
        "99.69" <-- our label/result
    ],
    [
        "62.89 / 50.31",
        "1.25"
    ],
    [
        "28.72 * 70.67 * -99.85 + 54.13 + 43.59",
        "-202562.07"
    ]
]
```

## Building a character tokenizer
Why a character tokenizer?

Our input is "10 + 5 * 3" and out put is "25". We have some known operators * / + -, and the digits 1234567890 positive and negative signs. We can't use word token or semi-word token because a number 123 is different from 321 or 124 or 12 or 23. Semi-number don't make sense, and mapping every number is not possible. Our embedding is discrete not continuos.

## Building a tiny transformer from scratch

This model will be a decoder only transformer that leverage "prefix language modeling" by allowing full attention over the input arithmetic expression and then autoregressively generating the numerical result.

Since this project is for learning puposes also, I am building every part of transformer from scratch.

## Causal masking

Since I need the model to see entire expression left of "=", my causal masking was only masking the result to right of "=".

## Computing the loss

Referncing this https://arxiv.org/pdf/2307.03381 paper again, very interestingly they used a trick where they reversed the result so that instead of the model learning 512 + 139 = 651, its 512 + 139 = 156. The author mentioned that because we do arithmetic digit-wise from right to left, we would typically write our solution from the right first, while mentally carrying over the number (if thats needed) to the left. Since transformers are autoregressive, we need to reverse the result.

Once the result is calculated, I apply a loss mask. Since our prefix inputs is full attention and sees every token, we dont want to calculate loss for prefix tokens. We also don't want to calculate loss for `<pad>` after our `<eos>`. This loss mask will produce a [0, 1] float dtype mask so we zero out the loss from tokens we dont care about, leaving only the loss updates for the output tokens and `<eos>`.

After loss mask is applied, I calculate the Cross Entropy Loss, pretty standard ... didn't see reason to change it.

## Positional Encoder

I know that some paper like https://arxiv.org/pdf/2307.03381 proposes using a specific position encoder that captures the decimal/digit position of a number, e.g - 6125 is encoded as 6 = 4, 1 = 3, 2 = 2, 5 = 1, to capture the number's digit from right to left, hinting to model that arithmetic starts with right most digit.

I chose to just use the OG 2017 sinusoidal PE instead because I want to learn building it from scratch, but also partially wanted to know if I could succeed without overengineering the model.

## Multihead Attention

This is pretty standard MHA, which I built from scratch to get the practice in. It does not have dropout because the model is a prefix language model, thus dropout does not make sense, the model needs to see the whole arithmetic expression left of "=".

## Curriculum learning

My hope was to train a fairly vanilla transformer model from scratch to be able to perform long sequence + - * / with basically just carefully crafted curriculum learning datasets. (This proved to be challenging ... woops)

The way I set up my curriculum is as below:

#### Start with addition (like in elementary school!)
- Integer only, [+], 1 to 2 operations, size = 50,000
- Integer only, [+], 1 to 5 operations, size = 100,000
- Integer + float mixed, [+], 1 to 2 operations, size = 200,000
- Integer + float mixed, [+], 1 to 5 operations, size = 300,000

#### Move on to subtraction
- Integer + float mixed, [+, -] with weight [40, 60], 1 to 2 operations, size = 300,000 
- Integer + float mixed, [+, -] with weight [10, 90], 1 to 5 operations, size = 300,000

#### Move on to multiplication
- Integer + float mixed, [+, -, *] with weight [45, 45, 10], 1 to 2 operations, size = 300,000
- Integer + float mixed, [+, -, *] with weight [33, 33, 34], 1 to 5 operations, size = 300,000

#### Move on to divide
- Integer + float mixed, [+, -, *, /] with weight [32, 32, 32, 4], 1 to 2 operations, size = 300,000
- Integer + float mixed, [+, -, *, /] with weight [30, 30, 30, 10], 1 to 2 operations, size = 300,000

#### All together
- Integer + float mixed, [+, -, *, /] with weight [25, 25, 25, 25], 1 to 2 operations, size = 300,000
- Integer + float mixed, [+, -, *, /] with weight [25, 25, 25, 25], 1 to 5 operations, size = 300,000

## Lessons learned (successes and failures)

1. Learning is fragile.... model overfitting or getting stuck (loss plateau) can happen easily at the beginning, when you're training from scratch, regardless of the model size, dataset size, learning rate, batch size...etc. I guess this is why pretraining phase is so important, and takes such a long time.

2. Just tossing a big dataset at a model and tell it to go learn does not work. This is obvious, but it was also interesting to see:

When using a complex dataset of + - * / that is long sequence, and mixed of int and float... even with 1mm samples dataset, and model config:
- batch_size = 512
- d_model = 512
- max_seq_len = 256
- num_heads = 8

It was producing gibberish... even at epoch 20, and loss is hitting wall at 1.5
```
Prompt: 2 - 2=
Model Output: -1.0.12.7

Prompt: 90 + 2 * 5=
Model Output: 13.01

Prompt: 11 + 12 + 13 * 10=
Model Output: 29.25

Prompt: 6 * 6 + 4=
Model Output: 41110.0

Prompt: 1 + 2 + 3 + 4 + 5=
Model Output: -2.88
```

3. The first 5-10 epoch is very important when training model from scratch. Reason = model undergoes rapid representation learning early on. This means that in the first few training epochs, the model can learn some strong pattern and lock in useful/harmful inductive bias.

Deep dive = At beginning, a model's weights are initialized using isotropic distribution. This means that the model's weights W is roughly full rank and it's singular values are uniform. Deep learning manifold hypothesis states that the input data X exist in a low dimensional continuous subspace called a manifold in a massive, high-dimensional ambient space $\mathbb{R}^n$. Here n is (N * d_model) of our model. So the goal here is to train our model, which begins with a representation that is full rank, to learn a representation that is low rank that closely approximates this manifold of input X. This means that during training, the model goes through gradual rank collapse. If my training data contains what the model perceives to be a "shortcut" or aka a highly reliable pattern that leads to lower CE loss. This can cause violent rank collapse.

4. Training a tranformer to learn + and - was quite straight forward, and did not require the 50k-300k sample size I used. I had a functional model for + and - at just:
- d_model = 512
- N = 256
- num_heads = 4
- layers = 2

with just 6.3mm params, and < 100k in total samples across all 6 curriculum datasets. The model was able to learn the rules of addition/subtraction and generalize to longer sequences as referenced in https://arxiv.org/pdf/2410.15787

5. Training a transformer to learn * was exceptionally hard... I tried every trick in the book. My conclusion was that, without specially engineering the model architecture for learning * or using chain of thoughts style learning, it is not possible for a model (regardless of size) to learn from input data alone. I could be wrong, maybe I just suck and can't figure it out... Conclusion = well curated curriculum learning is not enough for model to learn * using vanilla transformer model.


