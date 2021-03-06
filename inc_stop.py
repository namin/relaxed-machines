"""A differentiable machine that understands `STOP` and a subset of `INC`, `DEC`, `NOP`, `INC2`.

There is no data stack, just a data point.
There is also a program counter and a code bank.

Internally, the state also keeps track of whether the machine has halted,
to avoid running instructions after a STOP.
Without a `halted` state, the machine could get away with diffuse code,
counting while stopping before the needed incrementing.

The code parameterizes the neural network.
The parameters represent the code, and we learn the code.

The learning task is to learn counting (modulo n) by incrementing d times,
that is f(x)=(x+d) % n.
For n=5 and  d=3, we generate the examples
f(0) = 3, f(1) = 4, f(2) = 0, f(3) = 1, f(4) = 2
and the machine learns the program
INC, INC, INC, STOP, *
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

NOP = flags.DEFINE_boolean('nop', False, 'whether NOP is available as an instruction')
INC = flags.DEFINE_boolean('inc', True, 'whether INC is available as an instruction')
DEC = flags.DEFINE_boolean('dec', False, 'whether DEC is available as an instruction')
INC2 = flags.DEFINE_boolean('inc2', False, 'whether INC2 is available as an instruction')
INIT_NOP = flags.DEFINE_boolean('init_nop', False, 'whether to intialize with NOPs mostly as opposed to STOPs; only effective with --nop.')
N = flags.DEFINE_integer('n', 5, 'uniformly, number of integers and number of lines of code')
D = flags.DEFINE_integer('d', 3, 'learn f(x)=(x+d)%n')
SOFTMAX_SHARP = flags.DEFINE_float('softmax_sharp', 10, 'the multiplier to sharpen softmax')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, '')
TRAINING_STEPS = flags.DEFINE_integer('training_steps', 100000, '')
SEED = flags.DEFINE_integer('seed', 42, '')

def instruction_names():
    names = ['STOP']
    if INC.value:
        names.append('INC')
    if DEC.value:
        names.append('DEC')
    if INC2.value:
        names.append('INC2')
    if NOP.value:
        names.append('NOP')
    return names

class Machine(hk.RNNCore):
    def __init__(
            self,
            name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.n = N.value
        self.has_nop = NOP.value
        self.has_inc = INC.value
        self.has_dec = DEC.value
        self.has_inc2 = INC2.value
        self.init_nop = self.has_nop and INIT_NOP.value
        self.stop_matrix = jnp.identity(self.n)
        self.inc_matrix =  jnp.roll(jnp.identity(self.n), 1, axis=1)
        self.data_instructions = [self.stop_matrix]
        self.pc_instructions = [self.stop_matrix]
        if self.has_inc:
            self.data_instructions.append(self.inc_matrix)
            self.pc_instructions.append(self.inc_matrix)
        if self.has_dec:
            self.dec_matrix = jnp.transpose(self.inc_matrix)
            self.data_instructions.append(self.dec_matrix)
            self.pc_instructions.append(self.inc_matrix)
        if self.has_inc2:
            self.data_instructions.append(jnp.matmul(self.inc_matrix, self.inc_matrix))
            self.pc_instructions.append(self.inc_matrix)
        if self.has_nop:
            self.data_instructions.append(self.stop_matrix)
            self.pc_instructions.append(self.inc_matrix)
        self.ni = len(self.data_instructions)

    def __call__(self, inputs, prev_state):
        new_state = self.step(prev_state)
        new_data = new_state[0:self.n]
        new_halted = new_state[2*self.n:2*self.n+2]
        return (new_data, new_halted), new_state

    def initial_state(self, batch_size: Optional[int]):
        data = jnp.zeros([self.n]).at[0].set(1)
        pc = jnp.zeros([self.n]).at[0].set(1)
        halted = jnp.array([0, 1])
        state = jnp.concatenate((data, pc, halted))
        assert batch_size is None
        return state

    def step(self, state):
        code = self.get_code()
        data = jax.nn.softmax(SOFTMAX_SHARP.value*state[0:self.n])
        pc = jax.nn.softmax(SOFTMAX_SHARP.value*state[self.n:2*self.n])
        halted = jax.nn.softmax(SOFTMAX_SHARP.value*state[2*self.n:2*self.n+2])
        sel = jnp.zeros(self.ni)
        for i in range(self.n):
            sel += pc[i] * jax.nn.softmax(SOFTMAX_SHARP.value*code[i])
        data_instr = jnp.zeros([self.n,self.n])
        pc_instr = jnp.zeros([self.n,self.n])
        for i in range(self.ni):
            data_instr += sel[i] * self.data_instructions[i]
            pc_instr += sel[i] * self.pc_instructions[i]
        next_data = halted[0] * data + halted[1] * jnp.matmul(data, data_instr)
        next_pc = halted[0] * pc + halted[1] * jnp.matmul(pc, pc_instr)
        next_halted = jnp.array([halted[0] + halted[1]*sel[0], halted[1]*(1-sel[0])])
        next_state = jnp.concatenate((next_data, next_pc, next_halted))
        return next_state

    def get_code(self):
        return hk.get_parameter('code', [self.n, self.ni], init=self.make_code_fun())

    def make_code_fun(self):
        if self.init_nop:
            # all NOPs but last STOP
            code = jnp.array([[1.0 if (i==0 and (not self.has_nop or l==self.n-1)) or (i==self.ni-1 and (self.has_nop and l!=self.n-1)) else 0.0 for i in range(self.ni)] for l in range(self.n)])
        else:
            # all STOPs
            code = jnp.array([[1.0 if i==0 else 0.0 for i in range(self.ni)] for l in range(self.n)])

        def code_fun(shape, dtype):
            return code
        return code_fun

    def load_data(self, state, data):
        next_state = state.at[0:self.n].set(data)
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
  (logits, halted), _ = hk.dynamic_unroll(core, [jnp.zeros(core.n) for i in range(sequence_length)], initial_state)
  return (logits, halted)

def sequence_loss(t) -> jnp.ndarray:
  """Unrolls the network over a sequence of inputs & targets, gets loss."""
  # Note the [-1] is to consider only the final output, not the intermediary data points.
  (logits, halted) = forward(t['input'])
  log_probs = jax.nn.log_softmax(SOFTMAX_SHARP.value*logits[-1])
  log_probs_halted = jax.nn.log_softmax(SOFTMAX_SHARP.value*halted[-1])

  one_hot_labels = t['target'][-1]
  loss = -jnp.sum(one_hot_labels * log_probs)
  loss -= log_probs_halted[0]
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

def train_data_inc(d):
    r = []
    for i in range(N.value):
        data = jax.nn.one_hot(i, N.value)
        target = jax.nn.one_hot([(i+j+1)%N.value if j < d else (i+d)%N.value for j in range(N.value)], N.value)
        r.append({'input':data, 'target':target})
    return r

def to_discrete(a):
    return [jnp.argmax(x).item() for x in a]

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

    #print(state.params['machine']['code'])
    print('MACHINE CODE', 'for learning f(x)=(x+%d)%%%d' % (D.value, N.value))
    names = instruction_names()
    print([names[x]for x in to_discrete(state.params['machine']['code'])])

    _, forward_fn = hk.without_apply_rng(hk.transform(forward))
    for i in range(N.value):
        t = next(train_data)
        (logits, _) = forward_fn(state.params, t['input'])
        #print('input:', t['input'])
        #print(logits)
        print('input:', jnp.argmax(t['input']).item())
        print('output steps:', to_discrete(logits))

if __name__ == '__main__':
    app.run(main)
