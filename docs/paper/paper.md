---
title: "detequal: localizing run-to-run nondeterminism in PyTorch models"
tags:
  - Python
  - PyTorch
  - reproducibility
  - determinism
  - debugging
  - numerical computing
authors:
  - name: "TODO"
    orcid: "0000-0000-0000-0000"
    affiliation: 1
affiliations:
  - name: "TODO"
    index: 1
date: "TODO"
bibliography: paper.bib
---

# Summary

`detequal` is a PyTorch tool that finds and explains run-to-run nondeterminism
in a model. It runs a model multiple times with all random-number generators
fixed identically, hooks every module boundary, statistically separates genuine
behavioral divergence from ordinary floating-point round-off, and reports which
layer is nondeterministic and why, together with a suggested fix. The core is
architecture-agnostic and runs on any CPU or GPU; one isolated code path
characterizes the run-to-run determinism of native low-precision (NVFP4/MXFP4)
tensor cores available only on recent hardware, degrading to a clear
skip message elsewhere.

# Statement of need

Every PyTorch user who has run the same computation twice and obtained different
numbers has encountered nondeterminism. Its causes are individually documented —
atomic-reduction kernels, runtime kernel selection in cuDNN/cuBLAS, and
floating-point non-associativity — but existing resources tell a user only that
a *category* of operation is risky in general, never *which specific layer in
their specific model* is responsible.

Three established techniques bear on the problem — global determinism flags,
dispatch-level hooking, and statistical separation of numerical signal from
round-off noise — but they have not been composed into a tool that answers the
question a working researcher actually has: my model gives different results on
two runs; which layer, and why? `detequal` composes them and answers that
directly.

<!-- TODO: fold in the related-work differentiation (Hawkeye, Cim et al.,
TTrace, PyTorch-native tooling) once full-text review is complete. Credit
TTrace's statistical method explicitly as adapted, not reinvented. -->

# Functionality

<!-- TODO: describe the six modules briefly, referencing the validation table
(precision/recall/false-positive rate) once section 5.1 is populated. -->

# Design and reuse

`detequal` deliberately builds on the existing ecosystem rather than reinventing
it: the statistical significance test adapts a published method, and the
instrumentation builds on PyTorch's own hooking and debug machinery. This is a
design choice, stated here as a positive signal.

# Research use

<!-- TODO: cite the precision-ladder empirical study conducted with the tool as
the research-impact evidence. This must reference real, logged runs. -->

# Acknowledgements

<!-- TODO -->

# References
