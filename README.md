## Optimizing for the Future in Non-Stationary MDPs

This project contains code for the paper titled "Optimizing for the Future in Non-Stationary MDPs" that appeared at the Thirty-seventh International Conference on Machine Learning (ICML 2020) .

Pdf: [https://arxiv.org/abs/2005.08158](https://arxiv.org/abs/2005.08158) 


### Installation

This code is written in Python 3.6 with PyTorch 1.0.0


### How to run

`Environment` This folder contains the maze environment used in the paper.

`Src` This folder contains the code for the proposed and baseline algorithms.

`Swarm` This folder contains the code for performing hyper-parameter sweep


For a single run of any algorithm:

 `Src/NS_parser.py` First set the desired hyper-parameters in this file.
 
 `Src/run_NS.py` After that execute this file.
 
 Note: You might need to appropriately set the root of your project for the import commands to work.
 

### Bibliography

```bash
@article{chandak2020optimizing,
  title={Optimizing for the Future in Non-Stationary MDPs},
  author={Chandak, Yash and Theocharous, Georgios and Shankar, Shiv and Mahadevan, Sridhar and White, Martha and Thomas, Philip S},
  journal={Thirty-seventh International Conference on Machine Learning (ICML)},
  year={2020}
}
```` 

