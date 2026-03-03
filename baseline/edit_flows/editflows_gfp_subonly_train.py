#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backward-compatibility shim.

The monolithic trainer has been split into:
  - model.py  (shared model/tokenizer/sampler)
  - train.py  (training entrypoint)
  - sample.py (inference entrypoint)

This script forwards to train.py so existing commands keep working.
"""

from train import main


if __name__ == "__main__":
    main()
