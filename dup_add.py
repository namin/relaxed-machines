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

N = flags.DEFINE_integer('n', 5, 'uniformly, number of integers, number of lines of code and of data stack size')
A = flags.DEFINE_integer('a', 2, 'learn f(x)=(a*x+b)%n')
B = flags.DEFINE_integer('b', 0, 'learn f(x)=(a*x+b)%n')
M = flags.DEFINE_integer('m', 3, 'number of tests to evaluate after training')
P = flags.DEFINE_integer('p', -1, 'push instruction for each constant from 0 to argument')
SUBMASK = flags.DEFINE_boolean('submask', False, 'whether to mask all but the data stack')
SOFTMAX_SHARP = flags.DEFINE_float('softmax_sharp', 10, 'the multiplier to sharpen softmax')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, '')
TRAINING_STEPS = flags.DEFINE_integer('training_steps', 100000, '')
SEED = flags.DEFINE_integer('seed', 42, '')

def get_instruction_names(n, p):
    assert p <= n
    names = []
    if p>=0:
        names.extend([str(i) for i in range(p+1)])
    names.extend(['DUP', 'ADD', 'STOP'])
    return names

class InstructionSet:
    def __init__(self, n, p, s):
        self.n = n
        self.p = p
        self.s = s
        self.instruction_names = get_instruction_names(self.n, self.p)
        self.ni = len(self.instruction_names)
        self.index_STOP = self.ni-1
        self.stop_matrix = jnp.identity(self.n)
        self.inc_matrix =  jnp.roll(jnp.identity(self.n), 1, axis=1)
        self.dec_matrix = jnp.roll(jnp.identity(self.n), -1, axis=1)
        self.clear_data_value = jnp.zeros(self.n).at[0].set(1)
        assert self.is_instr('STOP', self.index_STOP)

    def push_instr_num(self, i):
        return jax.nn.one_hot(i, self.n)

    def is_push_instr(self, index):
        return self.p>=0 and index <= self.p

    def is_instr(self, name, index):
        return self.instruction_names[index] == name

    def push(self, data_p, data, data_value):
        next_data_p = jnp.matmul(data_p, self.inc_matrix)
        next_data = self.s.write_value(next_data_p, data, data_value)
        return (next_data_p, next_data)

    def pop(self, data_p, data):
        data_value = self.s.read_value(data_p, data)
        # Learning is easier when leaving the state dirty...
        next_data = data
        # next_data = self.s.write_value(data_p, data, self.clear_data_value)
        next_data_p = jnp.matmul(data_p, self.dec_matrix)
        return (next_data_p, next_data, data_value)

    def execute_instr(self, i, data_p, data, pc):
        if self.is_instr('STOP', i):
            return (data_p, data, pc)
        next_pc = jnp.matmul(pc, self.inc_matrix)
        if self.is_instr('DUP', i):
            data_value = self.s.read_value(data_p, data)
            (next_data_p, next_data) = self.push(data_p, data, data_value)
            return (next_data_p, next_data, next_pc)
        elif self.is_instr('ADD', i):
            (data_p, data, n1) = self.pop(data_p, data)
            (data_p, data, n2) = self.pop(data_p, data)
            nr = self.add(n1, n2)
            (data_p, data) = self.push(data_p, data, nr)
            return (data_p, data, next_pc)
        elif self.is_push_instr(i):
            (data_p, data) = self.push(data_p, data, self.push_instr_num(i))
            return (data_p, data, next_pc)

        # let an error be...

    def add(self, n1, n2):
        nr = jnp.zeros(self.n)
        m = jnp.identity(self.n)
        for i in range(self.n):
            nr += n2[i] * jnp.matmul(n1, m)
            m = jnp.matmul(m, self.inc_matrix)
        return nr

    def step(self, code, state):
        (data_p, data, pc, halted) = self.s.unpack(state, self.sm)
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

    def sm(self, x):
        return jax.nn.softmax(SOFTMAX_SHARP.value*x)

class DiscreteInstructionSet(InstructionSet):
    def __init__(self, n, p, s):
        super().__init__(n, p, s)

    def sm(self, x):
        return x

class MachineState:
    def __init__(self, n):
        self.n = n
        self.total = self.n*self.n+2*self.n+2

    def read_value(self, data_p, data):
        return jnp.matmul(data_p.T, data).T

    def write_value(self, data_p, data, data_value):
        old = jnp.outer(data_p, jnp.ones(self.n)) * data
        new = jnp.outer(data_p, data_value)
        return data - old + new

    def initial(self):
        data_p = jnp.zeros(self.n).at[0].set(1)
        data = jnp.zeros([self.n, self.n])
        for i in range(self.n):
            data = data.at[(i,0)].set(1)
        pc = jnp.zeros([self.n]).at[0].set(1)
        halted = jnp.array([0, 1])
        state = self.pack(data_p, data, pc, halted)
        return state

    def unpack(self, state, fn=lambda x: x):
        data_p = fn(self.data_p(state))
        data = self.over_data(self.data(state), fn)
        pc = fn(self.pc(state))
        halted = fn(self.halted(state))
        return (data_p, data, pc, halted)


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

    def mask(self, state):
        if SUBMASK.value:
            data_p = self.data_p(state)
            data = self.data(state)
            top = self.read_value(data_p, data)
            r = jnp.zeros(self.n*2)
            r = r.at[0:self.n].set(top)
            r = r.at[self.n:self.n*2].set(data_p)
            return r
        else:
            # Give up on on smaller state, and give out the whole state...
            return state

    def print(self, state):
        data_p = self.data_p(state)
        data = self.data(state)
        data_top_of_stack = self.read_value(data_p, data)
        pc = self.pc(state)
        halted = self.halted(state)

        data_p = to_discrete_item(data_p)
        data_top_of_stack = to_discrete_item(data_top_of_stack)
        data = to_discrete(data)
        pc = to_discrete_item(pc)
        halted = 'True ' if to_discrete_item(halted)==0 else 'False'

        print(f"""top: {data_top_of_stack}, pointer: {data_p}, pc: {pc}, halted: {halted}, data: {data}""")

class Machine(hk.RNNCore):
    def __init__(
            self,
            name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.n = N.value
        self.p = P.value
        self.s = MachineState(self.n)
        self.i = InstructionSet(self.n, self.p, self.s)
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

def code_for_lin(a, b, n, p):
    assert a > 0
    assert 2*(a-1)+1 + (2 if b>0 else 0) <= n
    assert p >= -1
    names = get_instruction_names(n, p)
    DUP_INDEX = 0+p+1
    ADD_INDEX = 1+p+1
    STOP_INDEX = 2+p+1
    code = jnp.zeros([n, len(names)])
    for i in range(a-1):
        code = code.at[(i,DUP_INDEX)].set(1)
    for i in range(a-1):
        code = code.at[(i+a-1,ADD_INDEX)].set(1)
    next = a-1+a-1
    if b>0:
        assert p > b
        code = code.at[(next, b)].set(1)
        next += 1
        code = code.at[(next, ADD_INDEX)].set(1)
        next += 1
    for i in range(next,n):
        code = code.at[(i,STOP_INDEX)].set(1)
    return code

def train_data_inc(a, b):
    n = N.value
    p = P.value
    code = code_for_lin(a, b, n, p)
    print('MACHINE CODE for training')
    print(discrete_code(code))
    i = DiscreteInstructionSet(n, p, MachineState(n))
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

def discrete_code(code):
    names = get_instruction_names(N.value, P.value)
    return [names[x]for x in to_discrete(code)]

def main(_):
    #flags.FLAGS([""])

    train_data = itertools.cycle(train_data_inc(A.value, B.value))

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
        print('MACHINE CODE learnt')
        print(discrete_code(state.params['machine']['code']))

    header()
    s = MachineState(N.value)
    _, forward_fn = hk.without_apply_rng(hk.transform(forward))
    for i in range(M.value):
        t = next(train_data)
        states = forward_fn(state.params, t['input'])
        print('input:', jnp.argmax(t['input']).item())
        for j, st in enumerate(states):
            s.print(st)
    header()

if __name__ == '__main__':
    app.run(main)
