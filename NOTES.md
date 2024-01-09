# Notes on Relaxed Machines

Learning is easier when leaving the state dirty, see `pop` in `sub`.

We use the same semantics for the discrete machine and the relaxed machine.
The only difference is the `sm` function, which is the identify function for the discrete case, and a massaged `softmax` function in the relaxed case.
