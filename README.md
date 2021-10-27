## Implicit Behavioral Cloning - PyTorch

Pytorch implementation of <a href="https://arxiv.org/abs/2109.00137">Implicit Behavioral Cloning</a>.

## Install

```bash
conda create -n ibc python=3.8
pip install -r requirements.txt
```

## Progress

* [x] Implement the coordinate regression task.
* [x] Implement an explicit policy.
* [x] Generate plot like in Figure 4.
* [x] Implement an implicit EBM model.
* [x] Implement stochastic optimizers.
    * [x] Derivative-free sampler.
    * [ ] Implement Langevin dynamics.

## Results

|             | Explicit Policy | Implicit Policy |
|-------------|-----------------|-----------------|
| 10 examples |<img src="assets/explicit_mse_10.png" width="300" height="200"/>|<img src="assets/implicit_ebm_10.png" width="300" height="200"/>|
| 30 examples |<img src="assets/explicit_mse_30.png" width="300" height="200"/>|<img src="assets/implicit_ebm_30.png" width="300" height="200"/>|

## Citation

```bibtex
@misc{florence2021implicit,
    title = {Implicit Behavioral Cloning},
    author = {Pete Florence and Corey Lynch and Andy Zeng and Oscar Ramirez and Ayzaan Wahid and Laura Downs and Adrian Wong and Johnny Lee and Igor Mordatch and Jonathan Tompson},
    year = {2021},
    eprint = {2109.00137},
    archivePrefix = {arXiv},
    primaryClass = {cs.RO}
}
```
