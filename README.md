# ppm-lm

A proof of concept for conversational AI (like GPT) using statistical compression. The attached dataset (`result10.txt`) is a trimmed version of the TinyStories dataset.

Example response to a prompt after 1min 40s of training (from scratch) on a 1MB slice of the TinyStories dataset:

```
User says: Once upon a time,
PAQ says:  there was a little boy named Timmy. Timmy liked to swim in the big ball how to catch it. Timmy that could fly. His mom took off.
```

Uses ppmd_sh, a variant of Dymitry Shkarin's PPMd var J modified by Eugene Shelwien. The PAQ variant uses PAQ8 code (chiefly) by Matt Mahoney.
