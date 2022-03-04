Next steps (TODOs)
----------

- [ ] Try a Transformer instead of the RNN.
- [ ] Add noise to gradient descent as suggested by the [Neural Programmer](https://arxiv.org/abs/1511.04834).
      Note: this might already be done with temperature.
- [ ] Implement sketches like ∂4.
- [ ] Implement optimizations such as symbolic execution of straight-line code (like ∂4, see section 3.4.1 of thesis). Note: I think this would only work for code in hard sketches. Would it help that much?
- [ ] Add batching.
- [ ] Debug order of training samples.
- [ ] Explore the approach of [Neural Programmer-Interpreters](https://arxiv.org/abs/1511.06279). It seems exciting because
> A trained NPI with fixed parameters and a learned library of programs, can act both as an interpreter and as a programmer. As an interpreter, it takes input in the form of a program embedding and input data and subsequently executes the program.  As a programmer, it uses samples drawn from a new task to generate a new program embedding that can be added to its library of programs.
- [ ] Explore [TerpreT](https://arxiv.org/abs/1608.04428) ([code](https://github.com/51alg/TerpreT)).
      Makes the language extensible by compiling to an intermediate representation, though this does not make it possible to learn a DSL itself, just DSL programs.

## Log

### [`inc_stop`](inc_stop.py)

- [x] Flesh out a baby machine, which can add by a constant using repeated `INC`.

- [x] Exploited a `halted` state to avoid running after stopping.

- [x] Read out the machine code symbolically at the end.

- [x] Added flags for selecting different subsets of the instruction set.

- [x] Implemented instructions `STOP`, `INC`, `DEC`, `INC2`, `NOP`.

- [ ] Consider refactoring the logic outside the network module as for `dup_add` to continue using this file as a starting point.

### [`dup_add`](dup_add.py)

- [x] Flesh out a machine that uses a stack.

- [x] Refactored most of the instruction and state logic to be outside the network module, to good effect.

- [x] Implemented principled masking over state, to avoid learning over entire state at each step.

- [x] Implemented instructions `STOP`, `DUP`, `ADD` and learning task to multiply by a constant.

- [x] Add instruction to push integer.

- [ ] The machine only learns well when giving it the data top of stack and pointer at each step.
      Feels like cheating.
      Even with that, the machine does not learn well at higher `a`s and `n`s. For example, `--a 4 --n 16`.
      Then, the machine needs the whole state, including the dirty bits due to popping.
      
- [ ] Would it help to penalize stack underflow?

### [`reg_jmp`](reg_jmp.py)

- [x] Flesh out a machine with labels and branching.

  - [x] A language with two registers (`A` and `B`) for defining addition by repeated incrementation.

- [x] Separate flags for number of integers and lines of code.

- [x] Wrote assertions to stand for tests.

- [x] Define sketches.
  - [x] Could implement a "soft" sketch as initialization.
  - [x] Could implement a "hard" sketch by restricting code parameters to holes.
        
    - [x] By itself, a hard sketch does not seem to help reduce training steps.
          However, masking is now possible!

- [ ] Unroll the number of steps according to the computation.

## [`sub`](sub.py)

- [x] Flesh out a machine with subroutines.

- [x] Learns garbage. Debug instructions. The machine is just clever, exploiting using the return stack out of bound.

- [ ] Investigate why learning improves when masks hide more.
      For example, with `--hard --sketch --n 5`, the machine finds the correct program with `--mask_pc --mask_data_p --mask_data --mask_ret_p mask_ret`
      but not when given the full state.
      Note that `--soft --sketch --n 5` succeeds with the later.

- [ ] Penalize unsafe code (with out-of-bound errors including dual-purposing labels as instructions).

- [x] Optionally use Gumbel-softmax instead of softmax. Required giving up on convenient `hk.without_apply_rng`.

- [x] Performance seems to be worse with the Gumbel-softmax option than without.

- [x] Used pytrees to improve the code (less repetition).
      However, due to the difference in how `softmax` items are grouped, we get worse performance.

- [x] Try incorporating temperature (inverse sharpness of softmax) decreasing during training.
      It makes learning worse.
