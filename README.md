# Efficient Python CERN STEAM Academy

Efficient Python is a practice-oriented course on building reliable, readable, and high-performance workflows for scientific computing. It begins by treating vectors, matrices, and tensors as computational array objects, with particular emphasis on shapes, axes, indexing, broadcasting, and memory-aware operations. Participants then learn how to turn arrays into inspectable numerical experiments using NumPy and Matplotlib, and explore the numerical foundations of machine-learning workflows, including data organization, model fitting, losses, gradients, scaling, stability, and validation.

The course then moves from individual calculations to maintainable programs and reproducible data pipelines, covering explicit data flow, function interfaces, project organization, robust file handling, and labelled analysis with pandas. Its final part develops a disciplined approach to performance: benchmark first, locate bottlenecks through profiling and memory inspection, and only then select an appropriate acceleration strategy. Topics include algorithmic improvement, vectorization, memory control, selective JIT compilation with Numba, and concurrent or parallel execution. Throughout the course, scientific correctness, transparency, and reproducibility remain as important as speed.

## Repository structure

```text
Examples-for-lectures/
Lecture-notes/
  Lecture_01/
  Lecture_02/
  ...
Practical-sessions/
  Exercises_01/
  Exercises_02/
  ...
```

* `Lecture-notes/` contains slides and additional materials discussed during lectures.
* `Examples-for-lectures/` contains examples linked directly from the lecture notes.
* `Practical-sessions/` contains notebooks and files for hands-on exercises.
