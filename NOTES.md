# NOTES

## KaMPIng

- KaMPIng does not support non-contiguous non-range data types.
  - e.g., sub-matrices

## MPL

- MPL supports sub-arrays (even multi-dimensional ones) and non-contiguous data types.
- However, work with the library is tedious.
- Doesn't natively support distributing into different layouts.

## Boost MPI

- Point-to-point communication can be made layout-agnostic via techniques such as serialization or flattening
  - can be optimized by constructing a skeleton of the data structure <!-- TODO skeleton is basically a list of offsets (and types) of all elements -->
- Doesn't natively support distributing non-contiguous data types. <!-- TODO or something like that; basically, the point is that scattering sub-matrices is nigh-impossible -->
