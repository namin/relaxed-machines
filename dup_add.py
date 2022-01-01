"""A differentiable machine that understands `DUP`, `ADD`, `STOP`.

The state of the machine comprises a data stack, a program counter and whether the machine has halted.

The parameters of the machine are the lines of code.

The learning task is to learn multiplication by d modulo n,
that is f(x)=(x*d) % n.
For n=5 and  d=2, we generate the examples
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

N = flags.DEFINE_integer('n', 5, 'uniformly, number of integers, number of lines of code and of data stack size')
D = flags.DEFINE_integer('d', 2, 'learn f(x)=(x*d)%n')
SOFTMAX_SHARP = flags.DEFINE_float('softmax_sharp', 10, 'the multiplier to sharpen softmax')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, '')
TRAINING_STEPS = flags.DEFINE_integer('training_steps', 100000, '')
SEED = flags.DEFINE_integer('seed', 42, '')

INSTRUCTION_NAMES = ['DUP', 'ADD', 'STOP']

class InstructionSet:
    def __init__(self, n, s):
        self.n = n
        self.s = s
        self.instruction_names = INSTRUCTION_NAMES
        self.index_STOP = 2
        self.ni = len(self.instruction_names)
        self.stop_matrix = jnp.identity(self.n)
        self.inc_matrix =  jnp.roll(jnp.identity(self.n), 1, axis=1)
        self.dec_matrix = jnp.roll(jnp.identity(self.n), -1, axis=1)
        assert self.is_instr('STOP', self.index_STOP)

    def read_value(self, data_p, data):
        return jnp.matmul(data_p.T, data).T

    def write_value(self, data_p, data, data_value):
        old = jnp.outer(data_p, jnp.ones(self.n)) * data
        new = jnp.outer(data_p, data_value)
        return data - old + new

    def is_instr(self, name, index):
        return self.instruction_names[index] == name

    def execute_instr(self, i, data_p, data, pc):
        if self.is_instr('STOP', i):
            return (data_p, data, pc)
        next_pc = jnp.matmul(pc, self.inc_matrix)
        if self.is_instr('DUP', i):
            data_value = self.read_value(data_p, data)
            next_data_p = jnp.matmul(data_p, self.inc_matrix)
            next_data = self.write_value(next_data_p, data, data_value)
            return (next_data_p, next_data, next_pc)
        elif self.is_instr('ADD', i):
            n1 = self.read_value(data_p, data)
            next_data_p = jnp.matmul(data_p, self.dec_matrix)
            n2 = self.read_value(next_data_p, data)
            nr = self.add(n1, n2)
            next_data = self.write_value(next_data_p, data, nr)
            return (next_data_p, next_data, next_pc)
        # let an error be...

    def add(self, n1, n2):
        nr = jnp.zeros(self.n)
        m = jnp.identity(self.n)
        for i in range(self.n):
            nr += n2[i] * jnp.matmul(n1, m)
            m = jnp.matmul(m, self.inc_matrix)
        return nr

    def step(self, code, state):
        data_p = self.sm(self.s.data_p(state))
        data = self.sm_over_data(self.s.data(state))
        pc = self.sm(self.s.pc(state))
        halted = self.sm(self.s.halted(state))
        sel = jnp.zeros(self.ni)
        for i in range(self.n):
            sel += pc[i] * self.sm(code[i])
        new_data_p = jnp.zeros(self.n)
        new_data = jnp.zeros([self.n, self.n])
        new_pc = jnp.zeros(self.n)
        for i in range(self.ni):
            (delta_data_p, delta_data, delta_pc) = self.execute_instr(i, data_p, data, pc)
            new_data_p += sel[i] * delta_data_p
            new_data += sel[i] * delta_data
            new_pc += sel[i] * delta_pc
        next_data_p = halted[0] * data_p + halted[1] * new_data_p
        next_data = halted[0] * data + halted[1] * new_data
        next_pc = halted[0] * pc + halted[1] * new_pc
        halting = sel[self.index_STOP]
        not_halting = 1-halting
        next_halted = jnp.array([halted[0] + halted[1]*halting, halted[1]*not_halting])
        next_state = self.s.pack(next_data_p, next_data, next_pc, next_halted)
        return next_state

    def sm_over_data(self, data):
        return self.s.over_data(data, self.sm)

    def sm(self, x):
        return jax.nn.softmax(SOFTMAX_SHARP.value*x)

    def print(self, state):
        # here rather than in MachineState so we can conveniently read from stack
        data_p = self.s.data_p(state)
        data = self.s.data(state)
        data_top_of_stack = self.read_value(data_p, data)
        pc = self.s.pc(state)
        halted = self.s.halted(state)

        data_p = to_discrete_item(data_p)
        data_top_of_stack = to_discrete_item(data_top_of_stack)
        data = to_discrete(data)
        pc = to_discrete_item(pc)
        halted = 'True ' if to_discrete_item(halted)==0 else 'False'

        print(f"""top: {data_top_of_stack}, pointer: {data_p}, pc: {pc}, halted: {halted}, data: {data}""")

class DiscreteInstructionSet(InstructionSet):
    def __init__(self, n, s):
        super().__init__(n, s)

    def sm(self, x):
        return x

class MachineState:
    def __init__(self, n):
        self.n = n
        self.total = self.n*self.n+2*self.n+2

    def initial(self):
        data_p = jnp.zeros(self.n).at[0].set(1)
        data = jnp.zeros([self.n, self.n])
        for i in range(self.n):
            data = data.at[(i,0)].set(1)
        pc = jnp.zeros([self.n]).at[0].set(1)
        halted = jnp.array([0, 1])
        state = self.pack(data_p, data, pc, halted)
        return state

    def pack(self, data_p, data, pc, halted):
        data = jnp.reshape(data, self.n*self.n)
        return jnp.concatenate((data_p, data, pc, halted))

    def initial_top_of_stack(self, state, data_value):
        # assumes the pointer is at the reset
        next_state = state.at[self.n:self.n*2].set(data_value)
        return next_state

    def data_p(self, state):
        return state[0:self.n]

    def data(self, state):
        data = state[self.n:self.n*self.n+self.n]
        return jnp.reshape(data, [self.n, self.n])

    def pc(self, state):
        return state[self.n*self.n+self.n:self.n*self.n+2*self.n]

    def halted(self, state):
        return state[self.n*self.n+2*self.n:self.n*self.n+2*self.n+2]

    def over_data(self, data, fn):
        for i in range(self.n):
            data = data.at[i].set(fn(data[i]))
        return data

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

    def initial_state(self, batch_size: Optional[int]):
        assert batch_size is None
        return self.s.initial()

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

    def load_data(self, state, data_value):
        next_state = self.s.initial_top_of_stack(state, data_value)
        return next_state

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
  initial_state = core.initial_state(batch_size=None)
  initial_state = core.load_data(initial_state, input)
  states, _ = hk.dynamic_unroll(core, [jnp.zeros(core.n) for i in range(sequence_length)], initial_state)
  return states

def sequence_loss(t) -> jnp.ndarray:
  """Unrolls the network over a sequence of inputs & targets, gets loss."""
  # We compute the loss over the ENTIRE state at each step...
  # ... definitely cheating.
  states = forward(t['input'])
  log_probs = jax.nn.log_softmax(SOFTMAX_SHARP.value*states)
  one_hot_labels = t['target']
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

def code_for_mul(d, n):
    assert d > 0
    assert 2*(d-1)+1 <= n
    code = jnp.zeros([n, len(INSTRUCTION_NAMES)])
    for i in range(d-1):
        code = code.at[(i,0)].set(1)
    for i in range(d-1):
        code = code.at[(i+d-1,1)].set(1)
    for i in range(d-1+d-1,n):
        code = code.at[(i,2)].set(1)
    return code

def train_data_inc(d):
    n = N.value
    code = code_for_mul(d, n)
    i = DiscreteInstructionSet(n, MachineState(n))
    r = []
    for j in range(N.value):
        data = jax.nn.one_hot(j, N.value)
        state = i.s.initial()
        state = i.s.initial_top_of_stack(state, data)
        target = []
        for k in range(N.value):
            state = i.step(code, state)
            target.append(state)
        r.append({'input':data, 'target': jnp.array(target)})
    return r

def to_discrete_item(x):
    return jnp.argmax(x).item()

def to_discrete(a):
    return [to_discrete_item(x) for x in a]

def main(_):
    #flags.FLAGS([""])

    train_data = itertools.cycle(train_data_inc(D.value))

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

    def header():
        print('MACHINE CODE', 'for learning f(x)=(x*%d)%%%d' % (D.value, N.value))
        names = INSTRUCTION_NAMES
        print([names[x]for x in to_discrete(state.params['machine']['code'])])

    header()
    instr_set = InstructionSet(N.value, MachineState(N.value))
    _, forward_fn = hk.without_apply_rng(hk.transform(forward))
    for i in range(N.value):
        t = next(train_data)
        states = forward_fn(state.params, t['input'])
        print('input:', jnp.argmax(t['input']).item())
        for j, s in enumerate(states):
            instr_set.print(s)
    header()

if __name__ == '__main__':
    app.run(main)
