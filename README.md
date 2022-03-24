Relaxed Machines
----------------

Explorations in neuro-symbolic differentiable interpreters.

## Inspired by ∂4

Differential interpreters in which the weights of the machine can be interpreted as a program.
I think this is different in spirit from ∂4, which corresponds to hard sketches here with holes that are whole neural networks.

Baby steps (see the [log](TODO.md#log)):

1. [`inc_stop`](inc_stop.py)
2. [`dup_add`](dup_add.py)
3. [`reg_jmp`](reg_jmp.py)
4. [`sub`](sub.py)

## Libraries
- [JAX](https://github.com/google/jax)
- [Haiku](https://github.com/deepmind/dm-haiku)
- [Optax](https://github.com/deepmind/optax)

## Resources
- Chapter 3 (∂4: A Differentiable Forth Interpreter) of Matko Bošnjak's Ph.D thesis, [On Differentiable Interpreters](https://discovery.ucl.ac.uk/id/eprint/10121772/), UCL, 2021.

## Acks

Many thanks to [Matko Bošnjak](https://matko.info/) and [Rob Zinkov](https://zinkov.com) for discussions, insights, suggestions, and pointers.
