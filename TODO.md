Next steps (TODOs)
----------

- [ ] Flesh out a machine with labels and branching.

  - [ ] A language with two registers (`A` and `B`) for defining addition by repeated increments.

Initially: `RA=n`, `RB=m`

Finally: `RA=0`, `RB=m+n`

Code:
```
0 JMP0_A 4
1 INC_B
2 DEC_A
3 JMP 0
4 STOP
```

- [ ] Flesh out a machine with subroutines.
- [ ] Define sketches.
  - [ ] Could implement a "soft" sketch as initialization.
  - [ ] Could implement a "hard" sketch by restricting code parameters to holes.
  - [ ] Could implement sketches like ∂4.
- [ ] Add batching.
- [ ] Write proper tests.

## Log

### [`inc_stop`](inc_stop.py)

- [x] Exploited a `halted` state to avoid running after stopping.

- [x] Read out the machine code symbolically at the end.

- [x] Added flags for selecting different subsets of the instruction set.

- [x] Implemented instructions `STOP`, `INC`, `DEC`, `INC2`, `NOP`.

- [x] Gave up on making uniform number flag non-uniform (e.g., separate for number of integers and number of lines of code).
      This confuses some instruction matrix operations that operate on data and pc.

- [ ] Consider refactoring the logic outside the network module as for `dup_add` to continue using this file as a starting point.

### [`dup_add`](dup_add.py)

- [x] Refactored most of the instruction and state logic to be outside the network module, to good effect.

- [x] Implemented principled masking over state, to avoid learning over entire state at each step.

- [x] Implemented instructions `STOP`, `DUP`, `ADD` and learning task to multiply by a constant.

- [x] Add instruction to push integer.

- [ ] The machine only learns well when giving it the data top of stack and pointer at each step.
      Feels like cheating.
      Even with that, the machine does not learn well at higher `a`s and `n`s. For example, `--a 4 --n 16`.
      Then, the machine needs the whole state, including the dirty bits due to popping.
      
- [ ] Would it help to penalize stack underflow?
