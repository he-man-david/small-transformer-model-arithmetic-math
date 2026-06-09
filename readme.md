# Training a small transformer to do arithmetics

A very fun project to mess around with tiny transformer models, and to better understand how a model do math.

## Creating our own dataset
Training data will be strings like "100 + 9 / 3", and the label will be "103". We will generate all our training data programmatically. To keep model small, we will keep length of expression short and only float up to 2 decimals.

Example:
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

## Building a character tokenizer
Why a character tokenizer?

Our input is "10 + 5 * 3" and out put is "25". We have some known operators * / + -, and the digits 1234567890 positive and negative signs. We can't use word token or semi-word token because a number 123 is different from 321 or 124 or 12 or 23. Semi-number don't make sense, and mapping every number is not possible. Our embedding is discrete not continuos.

## Building tiny transformer
Class for our tiny arithmetic transformer model. N = 256 d_model = 64 multiheads = 4

It will have 3 layers for FFN, with Relu activation and a layernorm pre-activation for each layer.

This model will be a decoder only transformer that leverage "prefix language modeling" by allowing full attention over the input arithmetic expression and then causally generating the numerical result.

Since this project is for learning puposes also, I am building every part of transformer from scratch.