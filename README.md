# SeeQ

## Introduction

SEEQ is "See Quantum" the Self-explanatory library for Quantum Optics and
Quantum Mechanics

It is a collection of Jupyter notebooks that use literary programming Python
implementation of traditional and modern algorithms in Quantum Mechanics and
Quantum Optics. For a related effort on Matrix Product States go to [SeeMPS](https://github.com/juanjosegarciaripoll/seemps).

The algorithms currently in SeeQ are:

* A pythonesque interface to evolve a quantum state with constant or
  time-dependent Hamiltonians.
* Algorithms to study stationary and time-dependent master equations.
* Chebyshev and Lanczos approximations to the exponentials of unitary matrices.
* An implementation of quantum control with parameterized pulses.

The main goal of this library is not performance, but rapid prototyping and
testing of ideas, providing a good playground before dwelling in more advanced
(C++, Julia) versions of the algorithms.

## Requirements

The library is entirely developed in Python 3 using Numpy and Scipy, and a
standard Jupyter environment. We recommend using Anaconda3 or Miniconda3,
although any other distribution of Python should suffice.

To rebuild the library you must have a copy of my `exportnb.py` utility together
with the `Makefile`. This utility is available from the
[exportnb](https://github.com/juanjosegarciaripoll/exportnb) repository.

## Usage

The library is explained in the self-contained notebooks with brief explanations
of the algorithms, the basic code, examples and tests. You can open the
notebooks and execute them. Each of them is self-contained, although it may rely
on other components of the library.

An index of the notebooks and of all algorithms is found in the [Introduction](00%20Introduction.ipynb)
notebook. This notebook also contains useful advice for using and modifying
the library.

The library can also be used as a standalone Python module, stored in the `seeq/`
directory. It consists of various submodules and functions that can be imported
as `import seeq`, `import seeq.evolution`, etc.

If you change the notebooks, the module `seeq/` must also be rebuilt. This is
done using `make all` at the toplevel directory. This works both in Windows as
well as in Linux/Mac (in Linux you may need to have `make` installed).

## License

This library is distributed as is, under a liberal license detailed in the file
`LICENSE`. However, if you use this material for research or educational purposes
or create derived works, we would like you to give credit to the library by
referencing the first work that we developed with it:

Adrian Feiguin, Juan José García Ripoll, Alejandro González Tudela,
"Qubit-Photon Corner States", arXiv

Alternatively, cite this work using its Zenodo reference (to be confirmed).

Version: 0.1

Authors:
* Juan José García Ripoll (Institute of Fundamental Physics)
