# Log

### Succeeded

- `python inc_stop.py`
- `python dup_add.py`
- `python reg_jmp.py`
- `python sub.py`
- `python sub.py --sketch --hard --sketch_no_jmp`
- `python sub.py --train_data_with_branch --sketch --hard`
- `python sub.py --train_data_with_branch --sketch --soft`
- `python sub.py --train_data_with_sub --sketch --hard`
- `python sub.py --train_data_with_sub --sketch --soft`
- `python sub.py --train_data_with_sub --sketch --soft --sketch_no_jmp --training_steps 100000`
- `python sub.py --hard --sketch --mask_pc --mask_data_p --mask_data --mask_ret_p mask_ret`
- `python sub.py --soft --sketch --mask_pc --mask_data_p --mask_data --mask_ret_p mask_ret`


## Failed

- `python sub.py --gumbel_softmax`
- `python sub.py --train_data_with_branch`
- `python sub.py --sketch --hard --sketch_no_jmp --training_steps 500000`
  (though it worked without the sketch, and though it worked with fewer training steps!)
- `python sub.py --train_data_with_sub --training_steps 5000000`
- `python sub.py --train_data_with_sub --sketch --hard --sketch_no_jmp --training_steps 500000`
