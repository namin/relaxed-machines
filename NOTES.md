# Notes on Relaxed Machines

The first experiment (`inc_stop`) use input/output examples, while the later ones (`dup_add`, `reg_jmp`, `sub`) use execution traces.
Execution traces started with `dup_add`, which only learns well when giving it the data top of stack and pointer at each step.

A hard sketch does not seem to help reduce training steps, but it makes masking possible.

Learning is easier when leaving the state dirty, see `pop` in `sub`.

We use the same semantics for the discrete machine and the relaxed machine.
The only difference is the `sm` function, which is the identify function for the discrete case, and a massaged `softmax` function in the relaxed case.
