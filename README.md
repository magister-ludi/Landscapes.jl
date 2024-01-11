# Landscapes.jl

This began as a Julia port of the Python code in
[landscapes](https://github.com/nathanrooy/landscapes).
This version has additional tests and plots of all the
functions that have reasonable 2D representations.

There is some overlap with the registered package
[OptimizationTestFunctions.jl](https://github.com/andrewjradcliffe/OptimizationTestFunctions.jl).
That package probably has faster implementations, where this package
has aimed for succinctness.

## Tests

The original [Python code](https://github.com/nathanrooy/landscapes)
has tests to confirm that function values at the extremal locations are
correct. Tests for this package also attempt, using
[BlackBoxOptim](https://github.com/robertfeldt/BlackBoxOptim.jl),
to demonstrate that the relevant locations really are minimal points.
These tests are simple and imprecise: comparisons in tests often permit quite large
differences. Tests in higher dimensions (≳ 5) are mostly skipped, as
greater care needs to be taken in order to demonstrate that
the locations really are minimums.

## Notebooks

The notebook files in https://github.com/magister-ludi/Landscapes.jl/tree/master/notebooks
provide plots of all the functions that are either two-dimensional or are
multi-dimensional, with a two-dimensional form.
