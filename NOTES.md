# Notes on Relaxed Machines

The earlier experiments (`inc_stop`, `dup_add`) use input/output examples, while the later ones (`reg_jmp`, `sub`) use execution traces.

Learning is easier when leaving the state dirty, see `pop` in `sub`.

We use the same semantics for the discrete machine and the relaxed machine.
The only difference is the `sm` function, which is the identify function for the discrete case, and a massaged `softmax` function in the relaxed case.
