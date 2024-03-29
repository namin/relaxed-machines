"""A differentiable machine with two registers, a data stack, a return stack, conditional and unconditional jumping, and subroutine calling and returning.

The parameters of the machine are the lines of code.
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

from tqdm import tqdm

import cache

import notify

PRNGKey = jnp.ndarray

NOTIFY = flags.DEFINE_boolean('notify', False, 'notify when training is complete (Mac OS X)')

N = flags.DEFINE_integer('n', 5, 'number of integers')
L = flags.DEFINE_integer('l', 10, 'number of lines of code')
M = flags.DEFINE_integer('m', 3, 'number of tests to evaluate after training')
S = flags.DEFINE_integer('s', 30, 'number of steps when running the machine')
SOFTMAX_SHARP = flags.DEFINE_float('softmax_sharp', 10, 'the multiplier to sharpen softmax (inverse temperature)')
GUMBEL_SOFTMAX = flags.DEFINE_boolean('gumbel_softmax', False, 'use Gumbel Softmax')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, '')
TRAINING_STEPS = flags.DEFINE_integer('training_steps', 60000, '')
SEED = flags.DEFINE_integer('seed', 42, '')

TRAIN_DATA_WITH_SUB = flags.DEFINE_boolean('train_data_with_sub', False, 'train the data with the _SUB program as opposed to the vanilla one')
TRAIN_DATA_WITH_BRANCH = flags.DEFINE_boolean('train_data_with_branch', False, 'train the data with the _BRANCH program as opposed to the vanilla one')
HARD_SKETCH = flags.DEFINE_boolean('hard', False, 'whether to use a hard sketch: only parameters for holes')
SOFT_SKETCH = flags.DEFINE_boolean('soft', False, 'whether to use a soft sketch: initial state, full parameters')
SKETCH = flags.DEFINE_boolean('sketch', False, 'whether to sketch')
SKETCH_NO_JMP = flags.DEFINE_boolean('sketch_no_jmp', False, 'whether to use a sketch that leaves holes for the jumps')
MASK_A = flags.DEFINE_boolean('mask_a', False, 'whether to mask A')
MASK_B = flags.DEFINE_boolean('mask_b', False, 'whether to mask B')
MASK_PC = flags.DEFINE_boolean('mask_pc', False, 'whether to mask PC')
MASK_HALTED = flags.DEFINE_boolean('mask_halted', False, 'whether to mask halted status')
MASK_DATA_P = flags.DEFINE_boolean('mask_data_p', False, 'whether to mask the data stack pointer')
MASK_DATA = flags.DEFINE_boolean('mask_data', False, 'whether to mask the data stack buffer')
MASK_RET_P = flags.DEFINE_boolean('mask_ret_p', False, 'whether to mask the return stack pointer')
MASK_RET = flags.DEFINE_boolean('mask_ret', False, 'whether to mask the return stack buffer')
FINAL = flags.DEFINE_boolean('final', False, 'whether to only learn on final (possibly masked) state')
REVEAL_FINAL = flags.DEFINE_boolean('reveal_final', False, 'whether to reveal final state')
CHECK_SIDE_BY_SIDE = flags.DEFINE_boolean('check_side_by_side', True, 'whether to check state side-by-side after training')
CHECK_END = flags.DEFINE_boolean('check_end', True, 'whether to check learned machine')

TEMPERATURE = flags.DEFINE_boolean('temperature', False, 'whether to use temperature in softmax calculation')
TEMPERATURE_INIT = flags.DEFINE_float('temperature_init', 8, 'initial temperature')

INSTRUCTION_NAMES = ['PUSH_A', 'PUSH_B', 'POP_A', 'POP_B', 'INC', 'INC_A', 'INC_B', 'DEC', 'DEC_A', 'DEC_B', 'JMP0', 'JMP0_A', 'JMP0_B', 'JMP', 'CALL', 'RET', 'NOP', 'STOP']
INSTRUCTION_MAP = dict([(instr, index) for index, instr in enumerate(INSTRUCTION_NAMES)])

ADD_BY_INC_SUB = [
    'JMP0_A', 6, # 0
    'CALL', 7,  # 2
    'JMP', 0,   # 4
    'STOP',     # 6
    'INC_B',    # 7: SUB
    'DEC_A',    # 8
    'RET'       # 9
]

ADD_BY_INC_SUB_SKETCH = [
    'JMP0_A', 6, # 0
    'CALL', 7,  # 2
    'JMP', 0,   # 4
    'STOP',     # 6
    'HOLE',#'INC_B',    # 7: SUB
    'HOLE',#'DEC_A',    # 8
    'RET'       # 9
]

ADD_BY_INC_SUB_SKETCH_NO_JMP = [
    'HOLE', 'HOLE',#'JMP0_A', 6, # 0
    'CALL', 7,  # 2
    'HOLE', 'HOLE',#'JMP', 0,   # 4
    'STOP',     # 6
    'INC_B',    # 7: SUB
    'DEC_A',    # 8
    'RET'       # 9
]

ADD_BRANCH = [
    'JMP0_B', 8,
    'JMP0_A', 8,
    'INC_B',
    'DEC_A',
    'JMP', 2,
    'STOP'
]

ADD_BRANCH_SKETCH = [
    'JMP0_B', 8,
    'JMP0_A', 8,
    'HOLE', #'INC_B',
    'HOLE', #'DEC_A',
    'JMP', 2,
    'STOP'
]

ADD_BRANCH_SKETCH_NO_JMP = [
    'HOLE', 'HOLE', #'JMP0_B', 8,
    'HOLE', 'HOLE', #'JMP0_A', 8,
    'INC_B',
    'DEC_A',
    'HOLE', 'HOLE', #'JMP', 2,
    'STOP'
]

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

def pick_program_suffix():
    return '_sub' if TRAIN_DATA_WITH_SUB.value else '_branch' if TRAIN_DATA_WITH_BRANCH.value else ''

def pick_ADD_BY_INC():
    return ADD_BY_INC_SUB if TRAIN_DATA_WITH_SUB.value else ADD_BRANCH if TRAIN_DATA_WITH_BRANCH.value else ADD_BY_INC

def pick_ADD_BY_INC_SKETCH():
    return ADD_BY_INC_SUB_SKETCH if TRAIN_DATA_WITH_SUB.value else ADD_BRANCH_SKETCH if TRAIN_DATA_WITH_BRANCH.value else ADD_BY_INC_SKETCH

def pick_ADD_BY_INC_SKETCH_NO_JMP():
    return ADD_BY_INC_SUB_SKETCH_NO_JMP if TRAIN_DATA_WITH_SUB.value else ADD_BRANCH_SKETCH_NO_JMP if TRAIN_DATA_WITH_BRANCH.value else ADD_BY_INC_SKETCH_NO_JMP

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
        self.temperature = TEMPERATURE.value
        if self.temperature:
            self.n_steps = TRAINING_STEPS.value
            self.temp_init = TEMPERATURE_INIT.value
            self.temp_fn = lambda step: 1+(self.temp_init-1)*(1-((step+1)/self.n_steps))
            self.temp_c = 1
            self.temp_a = self.temp_init - 1
            self.temp_d = 1e-5
            #self.temp_fn = lambda step: self.temp_c + self.temp_a*jnp.exp(-self.temp_d*step/self.n_steps)
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

    def push(self, buffer_p, buffer, buffer_value, buffer_size):
        next_buffer_p = jnp.matmul(buffer_p, self.inc_matrix(buffer_size))
        next_buffer = self.s.write_value(next_buffer_p, buffer, buffer_value, buffer_size)
        return (next_buffer_p, next_buffer)

    def pop(self, buffer_p, buffer, buffer_size):
        buffer_value = self.s.read_value(buffer_p, buffer)
        # Learning is easier when leaving the state dirty...
        next_buffer = buffer
        next_buffer_p = jnp.matmul(buffer_p, self.dec_matrix(buffer_size))
        return (next_buffer_p, next_buffer, buffer_value)

    def execute_instr(self, code, i, state):
        (reg_a, reg_b, pc, halted, data_p, data, ret_p, ret) = state
        instr = self.get_instruction_name(i)
        if instr == 'STOP':
            return (reg_a, reg_b, pc, halted, data_p, data, ret_p, ret)
        next_pc = jnp.matmul(pc, self.inc_matrix(self.l))
        if instr.startswith('PUSH'):
            assert instr == 'PUSH_A' or instr == 'PUSH_B'
            reg = reg_a if instr[-1] == 'A' else reg_b
            (data_p, data) = self.push(data_p, data, reg, self.n)
        elif instr.startswith('POP'):
            assert instr == 'POP_A' or instr == 'POP_B'
            (data_p, data, data_value) = self.pop(data_p, data, self.n)
            if instr[-1] == 'A':
                reg_a = data_value
            else:
                reg_b = data_value
        elif instr == 'INC' or instr == 'DEC':
            (data_p, data, data_value) = self.pop(data_p, data, self.n)
            if instr == 'INC':
                m = self.inc_matrix(self.n)
            else:
                assert instr == 'DEC'
                m = self.dec_matrix(self.n)
            data_value = jnp.matmul(data_value, m)
            (data_p, data) = self.push(data_p, data, data_value, self.n)
        elif instr == 'INC_A':
            reg_a = jnp.matmul(reg_a, self.inc_matrix(self.n))
        elif instr == 'INC_B':
            reg_b = jnp.matmul(reg_b, self.inc_matrix(self.n))
        elif instr == 'DEC_A':
            reg_a = jnp.matmul(reg_a, self.dec_matrix(self.n))
        elif instr == 'DEC_B':
            reg_b = jnp.matmul(reg_b, self.dec_matrix(self.n))
        elif instr == 'CALL':
            push_pc = jnp.matmul(next_pc, self.inc_matrix(self.l))
            (ret_p, ret) = self.push(ret_p, ret, push_pc, self.l)
            next_pc = jnp.matmul(next_pc, code)[0:self.l]
        elif instr == 'RET':
            (ret_p, ret, ret_value) = self.pop(ret_p, ret, self.l)
            next_pc = ret_value
        elif instr == 'JMP':
            next_pc = jnp.matmul(next_pc, code)[0:self.l]
        elif instr.startswith('JMP0'):
            assert instr == 'JMP0' or instr == 'JMP0_A' or instr == 'JMP0_B'
            if instr == 'JMP0':
                (data_p, data, data_value) = self.pop(data_p, data, self.n)
                reg = data_value
            else:
                reg = reg_a if instr[-1] == 'A' else reg_b
            p = jnp.dot(reg, jax.nn.one_hot(0, self.n))
            non_zero_next = jnp.matmul(next_pc, self.inc_matrix(self.l))
            zero_jmp = jnp.matmul(next_pc, code)[0:self.l]
            next_pc = (1-p)*non_zero_next + p*zero_jmp
        else:
            assert instr == 'NOP', f"Unhandled instruction {instr}"
        return (reg_a, reg_b, next_pc, halted, data_p, data, ret_p, ret)

    def step(self, code, state):
        (step, state) = state
        state = jax.tree_map(self.sm(step), state)
        pc = self.s.pc(state)
        halted = self.s.halted(state)
        sel = jnp.zeros(self.ni)
        for i in range(self.l):
            sel += pc[i] * self.sm(step)(code[i])
        new_state = jax.tree_map(lambda x: jnp.zeros(x.shape), state)
        for i in range(self.ni):
            delta_state = self.execute_instr(code, i, state)
            new_state = jax.tree_map(lambda new,delta: new + sel[i] * delta, new_state, delta_state)
        next_state = jax.tree_map(lambda old,new: halted[0] * old + halted[1] * new, state, new_state)
        halting = sel[self.index_STOP]
        not_halting = 1-halting
        next_halted = jnp.array([halted[0] + halted[1]*halting, halted[1]*not_halting])
        return (step+1, self.s.update_halted(next_state, next_halted))

    def sm(self, step):
        temp = self.temp_fn(step) if self.temperature else 1
        return logit_fn(jax.nn.softmax, temp)

    def empty_sketch(self):
        return ['HOLE' for i in range(self.l)]

    def is_wide_word(self, word):
        return isinstance(word, str) and word.startswith('JMP') or word == 'CALL'

    def fill_program(self, sketch, holes):
        hole_index = 0
        word_index = 0
        n_lines = len(sketch)
        n_holes = len(holes)
        program = []
        prev_word = None
        while word_index<n_lines:
            word = sketch[word_index]
            if word == 'HOLE':
                hole_item = to_discrete_item(holes[hole_index])
                if self.is_wide_word(prev_word):
                    hole_word = hole_item % self.l
                else:
                    hole_word = self.instruction_names[hole_item]
                word = hole_word
                program.append(word)
                hole_index += 1
            else:
                program.append(word)
            prev_word = word
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
            if self.is_wide_word(word) and index < self.l:
                w = to_discrete_item(code[index]) % self.l
                program.append(w)
                index += 1
        return program

    def enjolivate(self, program):
        return [str(i)+':'+('('+self.instruction_names[line]+'):'+str(line) if isinstance(line, int) else line) for i, line in enumerate(program)]

class DiscreteInstructionSet(InstructionSet):
    def __init__(self, n, l, s):
        super().__init__(n, l, s)

    def sm(self, step):
        return lambda x: x

class MachineState:
    def __init__(self, n, l):
        self.n = n
        self.l = l
        self.total = 2*self.n+self.l+2

    def read_value(self, p, buffer):
        return jnp.matmul(p.T, buffer).T

    def write_value(self, p, buffer, value, buffer_size):
        old = jnp.outer(p, jnp.ones(buffer_size)) * buffer
        new = jnp.outer(p, value)
        return buffer - old + new

    def new_stack(self, buffer_size):
        p = jnp.zeros(buffer_size).at[0].set(1)
        buffer = jnp.zeros([buffer_size, buffer_size])
        for i in range(buffer_size):
            buffer = buffer.at[(i,0)].set(1)
        return (p, buffer)

    def initial(self, reg_a, reg_b):
        pc = jnp.zeros([self.l]).at[0].set(1)
        halted = jnp.array([0.0, 1.0])
        (data_p, data) = self.new_stack(self.n)
        (ret_p, ret) = self.new_stack(self.l)
        state = (reg_a, reg_b, pc, halted, data_p, data, ret_p, ret)
        return (0, state)

    def reg_a(self, state):
        return state[0]

    def reg_b(self, state):
        return state[1]

    def pc(self, state):
        return state[2]

    def halted(self, state):
        return state[3]

    def update_halted(self, state, next_halted):
        mod_state = list(state)
        mod_state[3] = next_halted
        return tuple(mod_state)

    def data_p(self, state):
        return state[4]

    def data(self, state):
        return state[5]

    def ret_p(self, state):
        return state[6]

    def ret(self, state):
        return state[7]

    def discrete(self, state):
        return jax.tree_map(to_discrete_either, state)

    def check_similar_discrete(self, state1, state2):
        if CHECK_SIDE_BY_SIDE.value:
            (reg_a1, reg_b1, pc1, halted1, data_p1, data1, ret_p1, ret1) = self.discrete(state1)
            (reg_a2, reg_b2, pc2, halted2, data_p2, data2, ret_p2, ret2) = self.discrete(state2)
            if not MASK_A.value:
                assert reg_a1 == reg_a2, f"{reg_a1} should be == to {reg_a2}"
            if not MASK_B.value:
                assert reg_b1 == reg_b2, f"{reg_b1} should be == to {reg_b2}"
            if not MASK_PC.value:
                assert pc1 == pc2, f"{pc1} should be == to {pc2}"
            if not MASK_HALTED.value:
                assert halted1 == halted2, f"{halted1} should be == to {halted2}"
            if not MASK_DATA_P.value:
                assert data_p1 == data_p2, f"{data_p1} should be == to {data_p2}"
            if not MASK_DATA.value:
                assert data1 == data2, f"{data1} should be == to {data2}"
            if not MASK_RET_P.value:
                assert ret_p1 == ret_p2, f"{ret_p1} should be == to {ret_p2}"
            if not MASK_RET.value:
                assert ret1 == ret2, f"{ret1} should be == to {ret2}"

    def print(self, state):
        (reg_a, reg_b, pc, halted, data_p, data, ret_p, ret) = self.discrete(state)

        halted = 'True ' if halted==0 else 'False'

        print(f"""A: {reg_a}, B: {reg_b}, PC: {pc}, halted: {halted}""")
        print(f"""data: ({data_p}) {data}""")
        print(f"""ret: ({ret_p}) {ret}""")

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
                self.hard_sketch = pick_ADD_BY_INC_SKETCH_NO_JMP()
            else:
                self.hard_sketch = pick_ADD_BY_INC_SKETCH()
        else:
            self.hard_sketch = self.i.empty_sketch()
        self.init_hard_sketch()

    def __call__(self, inputs, prev_state):
        new_state = self.step(prev_state)
        return new_state[1], new_state

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
                code = self.i.sketch_to_one_hot(pick_ADD_BY_INC_SKETCH_NO_JMP())
            elif SKETCH.value:
                code = self.i.sketch_to_one_hot(pick_ADD_BY_INC_SKETCH())
            else:
                # we initialize to the whole program... a bit silly, but to try out
                code = self.i.sketch_to_one_hot(pick_ADD_BY_INC())
        else:
            # all holes are initialized to NOPs
            code = jnp.array([[1.0 if i==self.i.index_NOP else 0.0 for i in range(self.ni)] for l in range(self.n_holes)])

        def code_fun(shape, dtype):
            return code
        return code_fun

class DiscreteMachine:
    def __init__(self, i: InstructionSet, code):
        self.i = i
        self.code = code

    def __call__(self, inputs, prev_state):
        new_state = self.i.step(self.code, prev_state)
        return new_state[1], new_state

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

def mask():
    n = N.value
    l = L.value
    return (
        jnp.zeros(n) if MASK_A.value else jnp.ones(n),
        jnp.zeros(n) if MASK_B.value else jnp.ones(n),
        jnp.zeros(l) if MASK_PC.value else jnp.ones(l),
        jnp.zeros(2) if MASK_HALTED.value else jnp.ones(2),
        jnp.zeros(n) if MASK_DATA_P.value else jnp.ones(n),
        jnp.zeros([n,n]) if MASK_DATA.value else jnp.ones([n,n]),
        jnp.zeros(l) if MASK_RET_P.value else jnp.ones(l),
        jnp.zeros([l,l]) if MASK_RET.value else jnp.ones([l,l]),
    )

def logit_fn(softmax=jax.nn.softmax, temp=1.0):
    def fn(x):
        return softmax(SOFTMAX_SHARP.value*(1.0/temp)*(x+(jax.random.gumbel(hk.next_rng_key(), x.shape) if GUMBEL_SOFTMAX.value else 0)))
    return fn

def sequence_loss(t) -> jnp.ndarray:
  """Unrolls the network over a sequence of inputs & targets, gets loss."""
  states = forward(t['input'])
  log_probs = jax.tree_map(logit_fn(jax.nn.log_softmax), states)
  diffs = jax.tree_map(lambda x,y: x*y, log_probs, t['target'])
  diffs_masked = jax.tree_map(lambda x,y: x*y, diffs, mask())
  es, _ = jax.flatten_util.ravel_pytree(diffs_masked)
  n_items = len(t['target'])
  loss = 0
  factor = 1
  if REVEAL_FINAL.value or FINAL.value:
      factor = 2 if (not FINAL.value) else 1
      last_ones = jnp.array(jax.tree_map(lambda x: jnp.sum(x[-1]), diffs if REVEAL_FINAL.value else diffs_masked))
      loss += -jnp.sum(last_ones) / (n_items * factor)
  if not FINAL.value:
      loss += -jnp.sum(es) / (S.value * n_items * factor)
  return loss

@jax.jit
def update(state: TrainingState, rng_key: PRNGKey, t) -> TrainingState:
  """Does a step of SGD given inputs & targets."""
  _, optimizer = make_optimizer()
  _, loss_fn = hk.transform(sequence_loss)
  gradients = jax.grad(loss_fn)(state.params, rng_key, t)
  updates, new_opt_state = optimizer(gradients, state.opt_state)
  new_params = optax.apply_updates(state.params, updates)
  return TrainingState(params=new_params, opt_state=new_opt_state)

def check_add_by_inc(i, inp, fin):
    (reg_a, reg_b) = inp
    state = fin
    a = to_discrete_item(reg_a)
    b = to_discrete_item(reg_b)
    (res_a, res_b, res_pc, res_halted, res_data_p, res_data, res_ret_p, res_ret) = state
    d_a = to_discrete_item(res_a)
    d_b = to_discrete_item(res_b)
    d_halted = to_discrete_item(res_halted)
    assert d_halted == 0 # means halted...
    if not TRAIN_DATA_WITH_BRANCH.value:
        assert d_a == 0
        assert d_b == (a+b)%N.value, f'{d_b} vs ({a}+{b})%N'

def train_data_add_by_inc(n, l, s):
    #n = N.value
    #l = L.value
    #s = S.value
    program = pick_ADD_BY_INC()
    i = DiscreteInstructionSet(n, l, MachineState(n, l))
    code = i.program_to_one_hot(program)
    m = DiscreteMachine(i, code)
    print('MACHINE CODE for training')
    print(i.discrete_code(code))
    r = []
    for b in range(n):
        for a in range(n):
            print(f"A = {a}, B = {b}")
            reg_a = jax.nn.one_hot(a, n)
            reg_b = jax.nn.one_hot(b, n)
            initial_state = i.s.initial(reg_a, reg_b)
            states, _ = hk.dynamic_unroll(m, [jnp.zeros(s) for i in range(s)], initial_state)
            t = {'input':(reg_a, reg_b), 'target': states}
            check_add_by_inc(i, t['input'], [x[-1] for x in t['target']])
            r.append(t)
    return r

def to_discrete_item(x):
    return jnp.argmax(x).item()

def to_discrete(a):
    return [to_discrete_item(x) for x in a]

def to_discrete_either(x):
    if len(x.shape)==1:
        return to_discrete_item(x)
    else:
        return to_discrete(x)

def main(_):
    flags.FLAGS([""])

    rng = hk.PRNGSequence(SEED.value)

    train_data = cache.get_or_generate_data('data/inc'+pick_program_suffix()+'_%d_%d_%d.pickle', train_data_add_by_inc, N.value, L.value, S.value)
    n_train_data = len(train_data)

    train_data = itertools.cycle(train_data)
    # I tried shuffling the order but it made learning worse.
    def some_train_data(rng):
        #x = jax.random.randint(rng, [], 0, n_train_data)
        s = next(train_data)
        return s

    params_init, loss_fn = hk.transform(sequence_loss)
    opt_init, _ = make_optimizer()

    loss_fn = jax.jit(loss_fn)

    initial_params = params_init(next(rng), some_train_data(next(rng)))
    initial_opt_state = opt_init(initial_params)
    state = TrainingState(params=initial_params, opt_state=initial_opt_state)

    for step in tqdm(range(TRAINING_STEPS.value + 1)):
        t = some_train_data(next(rng))
        state = update(state, next(rng), t)

    if NOTIFY.value:
        notify.done()

    iset = InstructionSet(N.value, L.value, MachineState(N.value, L.value))
    holes = state.params['machine']['code']
    if HARD_SKETCH.value:
        if SKETCH_NO_JMP.value:
            learnt_program = iset.fill_program(pick_ADD_BY_INC_SKETCH_NO_JMP(), holes)
        elif SKETCH.value:
            learnt_program = iset.fill_program(pick_ADD_BY_INC_SKETCH(), holes)
        else:
            learnt_program = iset.discrete_code(holes)
    else:
        learnt_program = iset.discrete_code(holes)

    def header():
        print('MACHINE CODE learnt')
        print(learnt_program)
        print(iset.enjolivate(learnt_program))

    header()

    if not CHECK_END.value:
        return

    id = DiscreteInstructionSet(N.value, L.value, MachineState(N.value, L.value))
    idcode = id.program_to_one_hot(learnt_program)
    _, forward_fn = hk.transform(forward)
    s = S.value
    m = DiscreteMachine(id, idcode)
    for i in range(M.value):
        t = some_train_data(next(rng))
        inp = t['input']
        reg_a = inp[0]
        reg_b = inp[1]
        a = to_discrete_item(reg_a)
        b = to_discrete_item(reg_b)
        print('A:', a, ', B:', b)
        idstate = id.s.initial(reg_a, reg_b)
        states = forward_fn(state.params, next(rng), inp)
        states_discrete, _ = hk.dynamic_unroll(m, [jnp.zeros(s) for i in range(s)], idstate)
        t_states = per_state(states)
        t_states_discrete = per_state(states_discrete)
        two_states = map(lambda x,y: (x,y), t_states_discrete, t_states)
        halted = False
        for j, (idstate, st) in enumerate(two_states):
            iset.s.check_similar_discrete(idstate, st)
            new_halted  = to_discrete_item(iset.s.halted(st)) == 0
            if not halted:
                iset.s.print(st)
            else:
                assert halted == new_halted
            halted = new_halted
        check_add_by_inc(iset, inp, t_states[-1])
        check_add_by_inc(iset, inp, t_states_discrete[-1])

    header()

def per_state(t):
    return tree_transpose(jax.tree_map(lambda x: [el for el in x], t))

def tree_transpose(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of lists."""
    return jax.tree_map(lambda *xs: list(xs), *list_of_trees)

if __name__ == '__main__':
    app.run(main)
