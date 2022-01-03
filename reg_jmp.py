"""A differentiable machine with two registers and conditional and unconditional jumping.

The parameters of the machine are the lines of code.

The learning task is to learn addition by repeated incrementation.
"""

from typing import Any, NamedTuple, Optional

from absl import app
from absl import flags
from absl import logging

import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax

import itertools

N = flags.DEFINE_integer('n', 3, 'number of integers')
L = flags.DEFINE_integer('l', 9, 'number of lines of code')
M = flags.DEFINE_integer('m', 3, 'number of tests to evaluate after training')
S = flags.DEFINE_integer('s', 22, 'number of steps when running the machine')
SOFTMAX_SHARP = flags.DEFINE_float('softmax_sharp', 10, 'the multiplier to sharpen softmax')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, '')
TRAINING_STEPS = flags.DEFINE_integer('training_steps', 110000, '')
SEED = flags.DEFINE_integer('seed', 42, '')

HARD_SKETCH = flags.DEFINE_boolean('hard', False, 'whether to use a hard sketch: only parameters for holes')
SOFT_SKETCH = flags.DEFINE_boolean('soft', False, 'whether to use a soft sketch: initial state, full parameters')
SKETCH = flags.DEFINE_boolean('sketch', False, 'whether to sketch')
SKETCH_NO_JMP = flags.DEFINE_boolean('sketch_no_jmp', False, 'whether to use a hard sketch that leaves holes for the jumps')
MASK_A = flags.DEFINE_boolean('mask_a', False, 'whether to mask A')
MASK_B = flags.DEFINE_boolean('mask_b', False, 'whether to mask B')
MASK_PC = flags.DEFINE_boolean('mask_pc', False, 'whether to mask PC')
MASK_HALTED = flags.DEFINE_boolean('mask_halted', False, 'whether to mask halted status')
FINAL = flags.DEFINE_boolean('final', False, 'whether to only learn on final (possibly masked) state')

INSTRUCTION_NAMES = ['INC_A', 'INC_B', 'DEC_A', 'DEC_B', 'JMP0_A', 'JMP0_B', 'JMP', 'NOP', 'STOP']
INSTRUCTION_MAP = dict([(instr, index) for index, instr in enumerate(INSTRUCTION_NAMES)])

ADD_BY_INC = [
    'JMP0_A', 6,
    'INC_B',
    'DEC_A',
    'JMP', 0,
    'STOP'
]

ADD_BY_INC_SKETCH = [
    'JMP0_A', 6,
    'HOLE',#'INC_B',
    'HOLE',#'DEC_A',
    'JMP', 0,
    'STOP'
]

ADD_BY_INC_SKETCH_NO_JMP = [
    'HOLE', 'HOLE',#'JMP0_A', 6,
    'INC_B',
    'DEC_A',
    'HOLE', 'HOLE',#'JMP', 0,
    'STOP'
]

class InstructionSet:
    def __init__(self, n, l, s):
        self.n = n
        self.l = l
        self.s = s
        self.instruction_names = INSTRUCTION_NAMES
        self.instruction_map = INSTRUCTION_MAP
        self.ani = len(self.instruction_names)
        self.ni = self.ani
        if self.ni <= self.l:
            self.ni = self.l # pad to be able to address each of the l lines of code
        self.index_NOP = self.instruction_map['NOP']
        self.index_STOP = self.instruction_map['STOP']
        assert self.is_instr('STOP', self.index_STOP)

    def inc_matrix(self, dim, shift=1):
        return jnp.roll(jnp.identity(dim), shift, axis=1)

    def dec_matrix(self, dim):
        return self.inc_matrix(dim, -1)

    def get_instruction_name(self, index):
        if index < self.ani:
            return self.instruction_names[index]
        else:
            return 'NOP'

    def is_instr(self, name, index):
        return self.get_instruction_name(index) == name

    def execute_instr(self, code, i, reg_a, reg_b, pc):
        instr = self.get_instruction_name(i)
        if instr == 'STOP':
            return (reg_a, reg_b, pc)
        next_pc = jnp.matmul(pc, self.inc_matrix(self.l))
        if instr == 'INC_A':
            reg_a = jnp.matmul(reg_a, self.inc_matrix(self.n))
        elif instr == 'INC_B':
            reg_b = jnp.matmul(reg_b, self.inc_matrix(self.n))
        elif instr == 'DEC_A':
            reg_a = jnp.matmul(reg_a, self.dec_matrix(self.n))
        elif instr == 'DEC_B':
            reg_b = jnp.matmul(reg_b, self.dec_matrix(self.n))
        elif instr == 'JMP':
            next_pc = jnp.matmul(next_pc, code)[0:self.l]
        elif instr.startswith('JMP0_'):
            assert instr == 'JMP0_A' or instr == 'JMP0_B'
            reg = reg_a if instr[-1] == 'A' else reg_b
            p = jnp.dot(reg, jax.nn.one_hot(0, self.n))
            non_zero_next = jnp.matmul(next_pc, self.inc_matrix(self.l))
            zero_jmp = jnp.matmul(next_pc, code)[0:self.l]
            next_pc = (1-p)*non_zero_next + p*zero_jmp
        else:
            assert instr == 'NOP', f"Unhandled instruction {instr}"
        return (reg_a, reg_b, next_pc)

    def step(self, code, state):
        (reg_a, reg_b, pc, halted) = self.s.unpack(state, self.sm)
        sel = jnp.zeros(self.ni)
        for i in range(self.l):
            sel += pc[i] * self.sm(code[i])
        new_reg_a = jnp.zeros(self.n)
        new_reg_b = jnp.zeros(self.n)
        new_pc = jnp.zeros(self.l)
        for i in range(self.ni):
            (delta_reg_a, delta_reg_b, delta_pc) = self.execute_instr(code, i, reg_a, reg_b, pc)
            new_reg_a += sel[i] * delta_reg_a
            new_reg_b += sel[i] * delta_reg_b
            new_pc += sel[i] * delta_pc
        next_reg_a = halted[0] * reg_a + halted[1] * new_reg_a
        next_reg_b = halted[0] * reg_b + halted[1] * new_reg_b
        next_pc = halted[0] * pc + halted[1] * new_pc
        halting = sel[self.index_STOP]
        not_halting = 1-halting
        next_halted = jnp.array([halted[0] + halted[1]*halting, halted[1]*not_halting])
        next_state = self.s.pack(next_reg_a, next_reg_b, next_pc, next_halted)
        return next_state

    def sm(self, x):
        return jax.nn.softmax(SOFTMAX_SHARP.value*x)

    def empty_sketch(self):
        return ['HOLE' for i in range(self.l)]

    def fill_program(self, sketch, holes):
        hole_index = 0
        word_index = 0
        n_lines = len(sketch)
        n_holes = len(holes)
        program = []
        while word_index<n_lines:
            word = sketch[word_index]
            if word == 'HOLE':
                hole_item = to_discrete_item(holes[hole_index])
                hole_word = self.instruction_names[hole_item]
                program.append(hole_word)
                hole_index += 1
            else:
                program.append(word)
            word_index += 1
        assert hole_index == n_holes
        return program

    def sketch_to_one_hot(self, sketch):
        program = [word if word != 'HOLE' else 'NOP' for word in sketch]
        return self.program_to_one_hot(program)

    def program_to_one_hot(self, program):
        np = len(program)
        assert np <= self.l
        p = [self.instruction_map.get(word, word) for word in program]
        for i in range(np, self.l):
            p.append(self.index_STOP)
        r = jax.nn.one_hot(p, self.ni)
        return r

    def discrete_code(self, code):
        program = []
        index = 0
        while index < self.l:
            w = to_discrete_item(code[index])
            word = self.get_instruction_name(w)
            program.append(word)
            index += 1
            if word.startswith('JMP') and index < self.l:
                w = to_discrete_item(code[index])
                program.append(w)
                index += 1
        return program

class DiscreteInstructionSet(InstructionSet):
    def __init__(self, n, l, s):
        super().__init__(n, l, s)

    def sm(self, x):
        return x

class MachineState:
    def __init__(self, n, l):
        self.n = n
        self.l = l
        self.total = 2*self.n+self.l+2

    def initial(self, reg_a, reg_b):
        pc = jnp.zeros([self.l]).at[0].set(1)
        halted = jnp.array([0, 1])
        state = self.pack(reg_a, reg_b, pc, halted)
        return state

    def unpack(self, state, fn=lambda x: x):
        reg_a = fn(self.reg_a(state))
        reg_b = fn(self.reg_b(state))
        pc = fn(self.pc(state))
        halted = fn(self.halted(state))
        return (reg_a, reg_b, pc, halted)

    def pack(self, reg_a, reg_b, pc, halted):
        return jnp.concatenate((reg_a, reg_b, pc, halted))

    def reg_a(self, state):
        return state[0:self.n]

    def reg_b(self, state):
        return state[self.n:2*self.n]

    def pc(self, state):
        return state[2*self.n:2*self.n+self.l]

    def halted(self, state):
        return state[2*self.n+self.l:2*self.n+self.l+2]

    def mask(self, state):
        (reg_a, reg_b, pc, halted) = self.unpack(state)
        mask = []
        if not MASK_A.value:
            mask.append(reg_a)
        if not MASK_B.value:
            mask.append(reg_b)
        if not MASK_PC.value:
            mask.append(pc)
        if not MASK_HALTED.value:
            mask.append(halted)
        # weird: packing and unpacking the state degrades learning!
        return jnp.concatenate(mask)

    def print(self, state):
        reg_a = self.reg_a(state)
        reg_b = self.reg_b(state)
        pc = self.pc(state)
        halted = self.halted(state)

        reg_a = to_discrete_item(reg_a)
        reg_b = to_discrete_item(reg_b)
        pc = to_discrete_item(pc)
        halted = 'True ' if to_discrete_item(halted)==0 else 'False'

        print(f"""A: {reg_a}, B: {reg_b}, PC: {pc}, halted: {halted}""")

class Machine(hk.RNNCore):
    def __init__(
            self,
            name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.n = N.value
        self.l = L.value
        self.s = MachineState(self.n, self.l)
        self.i = InstructionSet(self.n, self.l, self.s)
        self.ni = self.i.ni
        if HARD_SKETCH.value:
            if SKETCH_NO_JMP.value:
                self.hard_sketch = ADD_BY_INC_SKETCH_NO_JMP
            else:
                self.hard_sketch = ADD_BY_INC_SKETCH
        else:
            self.hard_sketch = self.i.empty_sketch()
        self.init_hard_sketch()

    def __call__(self, inputs, prev_state):
        new_state = self.step(prev_state)
        return new_state, new_state

    def init_hard_sketch(self):
        self.hard_sketch_code = self.i.sketch_to_one_hot(self.hard_sketch)
        self.hole_indices = [i for (i, word) in enumerate(self.hard_sketch) if word == 'HOLE']
        self.n_holes = len(self.hole_indices)

    def initial_state(self, regs, batch_size: Optional[int]):
        assert batch_size is None
        (reg_a, reg_b) = regs
        return self.s.initial(reg_a, reg_b)

    def step(self, state):
        code = self.get_code()
        return self.i.step(code, state)

    def get_code(self):
        holes = hk.get_parameter('code', [self.n_holes, self.ni], init=self.make_code_fun())
        code = self.hard_sketch_code
        for i in range(self.n_holes):
            code = code.at[self.hole_indices[i]].set(holes[i])
        return code

    def make_code_fun(self):
        if SOFT_SKETCH.value:
            assert not HARD_SKETCH.value, "not yet supported"
            if SKETCH_NO_JMP.value:
                code = self.i.sketch_to_one_hot(ADD_BY_INC_SKETCH_NO_JMP)
            elif SKETCH.value:
                code = self.i.sketch_to_one_hot(ADD_BY_INC_SKETCH)
            else:
                # we initialize to the whole program... a bit silly, but to try out
                code = self.i.sletch_to_one_hot(ADD_BY_INC)
        else:
            # all holes are initialized to NOPs
            code = jnp.array([[1.0 if i==self.i.index_NOP else 0.0 for i in range(self.ni)] for l in range(self.n_holes)])

        def code_fun(shape, dtype):
            return code
        return code_fun

class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState

def make_network() -> Machine:
    model = Machine()
    return model

def make_optimizer() -> optax.GradientTransformation:
  """Defines the optimizer."""
  return optax.adam(LEARNING_RATE.value)

def forward(input) -> jnp.ndarray:
  core = make_network()
  sequence_length = S.value
  initial_state = core.initial_state(input, batch_size=None)
  states, _ = hk.dynamic_unroll(core, [jnp.zeros(sequence_length) for i in range(sequence_length)], initial_state)
  return states

def mask_each(xs):
    i = MachineState(N.value, L.value)
    return jnp.array([i.mask(x) for x in xs])

def sequence_loss(t) -> jnp.ndarray:
  """Unrolls the network over a sequence of inputs & targets, gets loss."""
  states = mask_each(forward(t['input']))
  log_probs = jax.nn.log_softmax(SOFTMAX_SHARP.value*states)
  one_hot_labels = mask_each(t['target'])
  if FINAL.value:
      one_hot_labels = one_hot_labels[-1]
      log_probs = log_probs[-1]
  loss = -jnp.sum(one_hot_labels * log_probs) / N.value
  return loss

@jax.jit
def update(state: TrainingState, t) -> TrainingState:
  """Does a step of SGD given inputs & targets."""
  _, optimizer = make_optimizer()
  _, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
  gradients = jax.grad(loss_fn)(state.params, t)
  updates, new_opt_state = optimizer(gradients, state.opt_state)
  new_params = optax.apply_updates(state.params, updates)
  return TrainingState(params=new_params, opt_state=new_opt_state)

def check_add_by_inc(i, inp, fin):
    (reg_a, reg_b) = inp
    state = fin
    a = to_discrete_item(reg_a)
    b = to_discrete_item(reg_b)
    (res_a, res_b, res_pc, res_halted) = i.s.unpack(state)
    d_a = to_discrete_item(res_a)
    d_b = to_discrete_item(res_b)
    d_halted = to_discrete_item(res_halted)
    assert d_halted == 0 # means halted...
    assert d_a == 0
    assert d_b == (a+b)%N.value, f'{d_b} vs ({a}+{b})%N'

def train_data_add_by_inc():
    n = N.value
    l = L.value
    program = ADD_BY_INC
    i = DiscreteInstructionSet(n, l, MachineState(n, l))
    code = i.program_to_one_hot(program)
    print('MACHINE CODE for training')
    print(i.discrete_code(code))
    r = []
    for a in range(n):
        for b in range(n):
            #print(f"A = {a}, B = {b}")
            reg_a = jax.nn.one_hot(a, n)
            reg_b = jax.nn.one_hot(b, n)
            state = i.s.initial(reg_a, reg_b)
            #i.s.print(state)
            target = []
            for k in range(S.value):
                state = i.step(code, state)
                #i.s.print(state)
                target.append(state)
            t = {'input':(reg_a, reg_b), 'target': jnp.array(target)}
            check_add_by_inc(i, t['input'], t['target'][-1])
            r.append(t)
    return r

def to_discrete_item(x):
    return jnp.argmax(x).item()

def to_discrete(a):
    return [to_discrete_item(x) for x in a]

def main(_):
    #flags.FLAGS([""])

    rng = hk.PRNGSequence(SEED.value)

    print('generating training data...')
    train_data = train_data_add_by_inc()
    n_train_data = len(train_data)
    print('...done!')

    train_data = itertools.cycle(train_data)
    # I tried shuffling the order but it made learning worse.
    def some_train_data(rng):
        #x = jax.random.randint(rng, [], 0, n_train_data)
        s = next(train_data)
        return s

    params_init, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
    opt_init, _ = make_optimizer()

    loss_fn = jax.jit(loss_fn)

    initial_params = params_init(next(rng), some_train_data(next(rng)))
    initial_opt_state = opt_init(initial_params)
    state = TrainingState(params=initial_params, opt_state=initial_opt_state)

    for step in range(TRAINING_STEPS.value + 1):
        t = some_train_data(next(rng))
        state = update(state, t)

    iset = InstructionSet(N.value, L.value, MachineState(N.value, L.value))
    holes = state.params['machine']['code']
    if HARD_SKETCH.value:
        if SKETCH_NO_JMP.value:
            learnt_program = iset.fill_program(ADD_BY_INC_SKETCH_NO_JMP, holes)
        elif SKETCH.value:
            learnt_program = iset.fill_program(ADD_BY_INC_SKETCH, holes)
        else:
            learnt_program = iset.discrete_code(holes)
    else:
        learnt_program = iset.discrete_code(holes)

    def header():
        print('MACHINE CODE learnt')
        print(learnt_program)

    header()
    _, forward_fn = hk.without_apply_rng(hk.transform(forward))
    for i in range(M.value):
        t = some_train_data(next(rng))
        inp = t['input']
        states = forward_fn(state.params, inp)
        check_add_by_inc(iset, inp, states[-1])
        print('A:', to_discrete_item(inp[0]), ', B:', to_discrete_item(inp[1]))
        halted = False
        for j, st in enumerate(states):
            new_halted  = to_discrete_item(iset.s.halted(st)) == 0
            if not halted:
                iset.s.print(st)
            else:
                assert halted == new_halted
            halted = new_halted

    header()

if __name__ == '__main__':
    app.run(main)
