"""A differentiable machine that understands `DUP`, `ADD`, `STOP`.

The state of the machine comprises a data stack, a program counter and whether the machine has halted.

The parameters of the machine are the lines of code.

The learning task is to learn a linear function,
that is f(x)=(a*x+b) % n.
For n=5, a=2, b=0, we generate the examples
f(0) = 0, f(1) = 2, f(2) = 4, f(3) = 1, f(4) = 3
and the machine learns the program
DUP, ADD, STOP, *
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

N = flags.DEFINE_integer('n', 7, 'uniformly, number of integers, number of lines of code and of data stack size')
M = flags.DEFINE_integer('m', 3, 'number of tests to evaluate after training')
SOFTMAX_SHARP = flags.DEFINE_float('softmax_sharp', 10, 'the multiplier to sharpen softmax')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, '')
TRAINING_STEPS = flags.DEFINE_integer('training_steps', 100000, '')
SEED = flags.DEFINE_integer('seed', 42, '')

INSTRUCTION_NAMES = ['INC_B', 'DEC_A', 'JMP0_A', 'JMP', 'STOP']
INSTRUCTION_MAP = dict([(instr, index) for index, instr in enumerate(INSTRUCTION_NAMES)])

class InstructionSet:
    def __init__(self, n, s):
        self.n = n
        self.s = s
        self.instruction_names = INSTRUCTION_NAMES
        self.instruction_map = INSTRUCTION_MAP
        self.ani = len(self.instruction_names)
        assert self.ani <= self.n
        self.ni = self.n # pad to be able to address each of the n lines of code
        self.index_STOP = self.instruction_map['STOP']
        self.stop_matrix = jnp.identity(self.n)
        self.inc_matrix =  jnp.roll(jnp.identity(self.n), 1, axis=1)
        self.dec_matrix = jnp.roll(jnp.identity(self.n), -1, axis=1)
        assert self.is_instr('STOP', self.index_STOP)

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
        next_pc = jnp.matmul(pc, self.inc_matrix)
        if instr == 'INC_B':
            reg_b = jnp.matmul(reg_b, self.inc_matrix)
        elif instr == 'DEC_A':
            reg_a = jnp.matmul(reg_a, self.dec_matrix)
        elif instr == 'JMP':
            next_pc = jnp.matmul(next_pc, code)
        elif instr == 'JMP0_A':
            p = jnp.dot(reg_a, jax.nn.one_hot(0, self.n))
            next_pc = (1-p)*next_pc + p*jnp.matmul(next_pc, code)
        else:
            assert instr == 'NOP'
        return (reg_a, reg_b, next_pc)

    def step(self, code, state):
        (reg_a, reg_b, pc, halted) = self.s.unpack(state, self.sm)
        sel = jnp.zeros(self.ni)
        for i in range(self.n):
            sel += pc[i] * self.sm(code[i])
        new_reg_a = jnp.zeros(self.n)
        new_reg_b = jnp.zeros(self.n)
        new_pc = jnp.zeros(self.n)
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

    def program_to_one_hot(self, program):
        np = len(program)
        assert np <= self.n
        p = [self.instruction_map.get(word, word) for word in program]
        for i in range(np, self.n):
            p.append(self.index_STOP)
        r = jax.nn.one_hot(p, self.n)
        return r

    def discrete_code(self, code):
        program = []
        index = 0
        while index < self.n:
            w = to_discrete_item(code[index])
            word = self.get_instruction_name(w)
            program.append(word)
            index += 1
            if word.startswith('JMP') and index < self.n:
                w = to_discrete_item(code[index])
                program.append(w)
                index += 1
        return program

class DiscreteInstructionSet(InstructionSet):
    def __init__(self, n, s):
        super().__init__(n, s)

    def sm(self, x):
        return x

class MachineState:
    def __init__(self, n):
        self.n = n
        self.total = 3*self.n+2

    def initial(self, reg_a, reg_b):
        pc = jnp.zeros([self.n]).at[0].set(1)
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
        return state[2*self.n:3*self.n]

    def halted(self, state):
        return state[3*self.n:3*self.n+2]

    def mask(self, state):
        return state

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
        self.s = MachineState(self.n)
        self.i = InstructionSet(self.n, self.s)
        self.ni = self.i.ni

    def __call__(self, inputs, prev_state):
        new_state = self.step(prev_state)
        return new_state, new_state

    def initial_state(self, regs, batch_size: Optional[int]):
        assert batch_size is None
        (reg_a, reg_b) = regs
        return self.s.initial(reg_a, reg_b)

    def step(self, state):
        code = self.get_code()
        return self.i.step(code, state)

    def get_code(self):
        return hk.get_parameter('code', [self.n, self.ni], init=self.make_code_fun())

    def make_code_fun(self):
        # all STOPs
        code = jnp.array([[1.0 if self.i.is_instr('STOP', i) else 0.0 for i in range(self.ni)] for l in range(self.n)])

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
  sequence_length = core.n
  initial_state = core.initial_state(input, batch_size=None)
  states, _ = hk.dynamic_unroll(core, [jnp.zeros(core.n) for i in range(sequence_length)], initial_state)
  return states

def mask_each(xs):
    i = MachineState(N.value)
    return jnp.array([i.mask(x) for x in xs])

def sequence_loss(t) -> jnp.ndarray:
  """Unrolls the network over a sequence of inputs & targets, gets loss."""
  states = mask_each(forward(t['input']))
  log_probs = jax.nn.log_softmax(SOFTMAX_SHARP.value*states)
  one_hot_labels = mask_each(t['target'])
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

ADD_BY_INC = [
    'JMP0_A', 6,
    'INC_B',
    'DEC_A',
    'JMP', 0,
    'STOP'
]

def train_data_inc():
    n = N.value
    program = ADD_BY_INC
    i = DiscreteInstructionSet(n, MachineState(n))
    code = i.program_to_one_hot(program)
    print('MACHINE CODE for training')
    print(i.discrete_code(code))
    r = []
    for a in range(N.value):
        for b in range(M.value):
            reg_a = jax.nn.one_hot(a, N.value)
            reg_b = jax.nn.one_hot(b, N.value)
            state = i.s.initial(reg_a, reg_b)
            target = []
            for k in range(N.value):
                state = i.step(code, state)
                target.append(state)
            r.append({'input':(reg_a, reg_b), 'target': jnp.array(target)})
    return r

def to_discrete_item(x):
    return jnp.argmax(x).item()

def to_discrete(a):
    return [to_discrete_item(x) for x in a]

def main(_):
    #flags.FLAGS([""])

    train_data = itertools.cycle(train_data_inc())

    params_init, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
    opt_init, _ = make_optimizer()

    loss_fn = jax.jit(loss_fn)

    rng = hk.PRNGSequence(SEED.value)
    initial_params = params_init(next(rng), next(train_data))
    initial_opt_state = opt_init(initial_params)
    state = TrainingState(params=initial_params, opt_state=initial_opt_state)

    for step in range(TRAINING_STEPS.value + 1):
        t = next(train_data)
        state = update(state, t)

    iset = InstructionSet(N.value, MachineState(N.value))
    def header():
        print('MACHINE CODE learnt')
        print(iset.discrete_code(state.params['machine']['code']))

    header()
    _, forward_fn = hk.without_apply_rng(hk.transform(forward))
    for i in range(M.value):
        t = next(train_data)
        states = forward_fn(state.params, t['input'])
        inp = t['input']
        print('A:', to_discrete_item(inp[0]), ', B:', to_discrete_item(inp[1]))
        for j, st in enumerate(states):
            iset.s.print(st)
    header()

if __name__ == '__main__':
    app.run(main)
