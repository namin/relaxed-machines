Next steps (TODOs)
----------

- [ ] Flesh out a machine with a proper data stack and more instructions.
  - [x] Instructions `DUP`, `ADD`, enough to multiply by a constant.
    - [x] Avoid giving away entire state for at each step for learning. Experiment with masks.
          Able to learn with a mask of data stack pointer and top of stack.
  - [ ] Add instruction to push integer.
- [ ] Flesh out a machine with labels and branching.
- [ ] Flesh out a machine with subroutines.
- [ ] Define sketches.
  - [ ] Could implement a "soft" sketch as initialization.
  - [ ] Could implement a "hard" sketch by restricting code parameters to holes.
  - [ ] Could implement sketches like ∂4.
- [ ] Write proper tests.

## Done

- [x] Avoid the RNN artifact of having to specify all the intermediate data points.
      Need to encode that any instruction after stop is ignored.
      Done by adding `halted` to state.

- [x] The to_discrete function needs to take into account all the positions that could be one instruction.

- [x] Perhaps, no need for number flag n to be uniform.
      Not needed for instructions, but needed otherwise
      because of the uniform way instruction matrices operate on data and pc.

- [x] Make the machine code symbolic at the end.
