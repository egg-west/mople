Eliminate flash_attention under V100 nodes
```
deepspeed train_rm.py --configs defaults_rm hh-data --use_flash_attention=false
```