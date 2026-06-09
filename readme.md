# Training a small transformer to do arithmetics

A very fun project to mess around with tiny transformer models, and to better understand how a model do math.

## Building a character tokenizer¶
Why a character tokenizer?

Our input is "10 + 5 * 3" and out put is "25". We have some known operators * / + -, and the digits 1234567890 positive and negative signs. We can't use word token or semi-word token because a number 123 is different from 321 or 124 or 12 or 23. Semi-number don't make sense, and mapping every number is not possible. Our embedding is discrete not continuos.

