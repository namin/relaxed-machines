Next steps (TODOs)
----------

- [ ] Flesh out a machine with a proper stack and more instructions.
- [ ] Define sketches.
- [ ] Make the machine code symbolic at the end.
- [ ] Write proper tests.
- [ ] Perhaps, no need for number flag n to be uniform.

## Done

- [x] Avoid the RNN artifact of having to specify all the intermediate data points.
      Need to encode that any instruction after stop is ignored.
      Done by adding `halted` to state.

- [x] The to_discrete function needs to take into account all the positions that could be one instruction.
