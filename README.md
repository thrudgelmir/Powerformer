# Installation Instructions

**Note:** This project uses Git LFS (Large File Storage) to manage large files.
To properly download all files, please install Git LFS before cloning or pulling:

```bash
# Install Git LFS (only once per system)
git lfs install
# Then clone the repository as usual
git clone <repo_url>
```

To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```

# Project Overview
This project includes Experiments in section 6.

## Running the Program

1. Plaintext Evaluation

To see plaintext experiments, run
- `tests/BPMax_Parameter_p,c.ipynb`: Study of BPMax parameter p,c (secsion 6.1.1)
- `tests/BatchLN_Parameter_l.ipynb`: Study of BatchLN parameter l (secsion 6.1.2)
- `tests/Training_Stretegy.ipynb`: Study of Training Strategy (secsion 6.1.3)
- `tests/Batch_Method.ipynb`: Study of Batch Method (secsion 6.1.4)

2. Ciphertext Evaluation
   
To see the homomorphic encryption experiment, run `./powerformer_HE/test.ipynb`.

Since a private GPU-based HE library is used, the results displayed are implemented in numpy. To obtain actual homomorphic encryption results, override the basic homomorphic operations in `./powerformer_HE/base_fncs.py` to match the functions of the HE library you intend to use.

## Execution
This project's code is provided in Jupyter Notebook (`.ipynb`) format.
