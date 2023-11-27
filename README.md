Eliminate flash_attention under V100 nodes
```
# you need python 3.10 to run this code
deepspeed train_rm.py --configs defaults_rm hh-data --use_flash_attention=false
deepspeed train_multi_head.py --configs defaults_rm oasst-rm-1-pythia-1.4b-multi-head --use_flash_attention=false
deepspeed train_llama2_multiHead.py --configs defaults_rm oasst-rm-1-pythia-1.4b-multi-head --use_flash_attention=false
```