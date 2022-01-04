Next steps (TODOs)
----------

- [ ] Learn whether pytrees can improve the code (less repetition).
- [ ] Implement sketches like ∂4.
- [ ] Add batching.
- [ ] Debug order of training samples.

## Log

### [`inc_stop`](inc_stop.py)

- [x] Flesh a baby machine, which can add by a constant using repeated `INC`.

- [x] Exploited a `halted` state to avoid running after stopping.

- [x] Read out the machine code symbolically at the end.

- [x] Added flags for selecting different subsets of the instruction set.

- [x] Implemented instructions `STOP`, `INC`, `DEC`, `INC2`, `NOP`.

- [ ] Consider refactoring the logic outside the network module as for `dup_add` to continue using this file as a starting point.

### [`dup_add`](dup_add.py)

- [x] Flesh a machine that uses a stack.

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

- [ ] Instead of
      ```['JMP0_A', 6, 'INC_B', 'DEC_A', 'JMP', 0, 'STOP']```
      the machine learns
      ```['JMP0_A', 6, 'INC_B', 'DEC_A', 'JMP0_A', 1, 'STOP']```
      which is correct, and even slightly more efficient (in terms of number of steps).
      Weirdly, this happens when using a full mask vs the state, which shouldn't make a difference.

- [ ] Unroll the number of steps according to the computation.

## [`sub`](sub.py)

- [x] Flesh out a machine with subroutines.

- [ ] Learns garbage. Debug instructions.
