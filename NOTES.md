# Notes on Relaxed Machines

## Methodology

### ∂4 primer

We take from ∂4 the relaxation of a machine.

### How we represent programs and programs with holes

A program has a fixed max number of lines of code, _l_.
Each line of code represents an instruction or an address to a line, and can be a _HOLE_.
The *HOLE*s are turned into parameters of the neural network (RNN).

In a hard sketch, some of the lines of code have fixed instructions and others have holes.
We can start with an empty sketch by putting a _HOLE_ for each line.
In a soft sketch, a given line of code is still a _HOLE_ but initialized with a given value.

An instruction is represented by a 1-hot encoding of size _ni_, the number of instructions, which is padded to at least _l_ to be able to address each of the _l_ lines of code.

### How we run it for synthesis

### How we train from data

### How we generate that data

## Notes

The first experiment (`inc_stop`) use input/output examples, while the later ones (`dup_add`, `reg_jmp`, `sub`) use execution traces.
Execution traces started with `dup_add`, which only learns well when giving it the data top of stack and pointer at each step.

A hard sketch does not seem to help reduce training steps, but it makes masking possible.

Learning is easier when leaving the state dirty, see `pop` in `sub`.

We use the same semantics for the discrete machine and the relaxed machine.
The only difference is the `sm` function, which is the identify function for the discrete case, and a massaged `softmax` function in the relaxed case.
