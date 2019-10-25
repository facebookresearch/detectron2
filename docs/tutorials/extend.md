# Extend Detectron2's Defaults

__Research is about doing things in new ways__.
This brings a tension in how to create abstractions in code,
which is a challenge for any research engineering project of a significant size:

1. On one hand, it needs to have very thin abstractions to allow for the possibility of doing
   everything in new ways. It should be reasonably easy to break existing
   abstractions and replace them with new ones.

2. On the other hand, such a project also needs reasonably high-level
   abstractions, so that users can easily do things in standard ways,
   without worrying too much about the details that only certain researchers care about.

In detectron2, there are two types of interfaces that address this tension together:

1. Functions and classes that take only a "config" argument (optionally with a minimal
   set of extra arguments in cases of mature interfaces).

   Such functions and classes implement
   the "standard default" behavior: it will read what it needs from the
   config and do the "standard" thing.
   Users only need to load a standard config and pass it around, without having to worry about
   which arguments are used and what they all mean.

2. Functions and classes that have well-defined explicit arguments.

   Each of these is a small building block of the entire system.
   They require users' effort to stitch together, but can be stitched together in more flexible ways.
   When you need to implement something different from the "standard defaults"
   included in detectron2, these well-defined components can be reused.


If you only need the standard behavior, the [Beginner's Tutorial](getting_started.html)
should suffice. If you need to extend detectron2 to your own needs,
see the following tutorials for more details:

* Detectron2 includes a few standard datasets, but you can use custom ones. See
  [Use Custom Datasets](datasets.html).
* Detectron2 contains the standard logic that creates a data loader from a
  dataset, but you can write your own as well. See [Use Custom Data Loaders](data_loading.html).
* Detectron2 implements many standard detection models, and provide ways for you
  to overwrite its behaviors. See [Use Models](models.html) and [Write Models](write-models.html).
* Detectron2 provides a default training loop that is good for common training tasks.
  You can customize it with hooks, or write your own loop instead. See [training](training.html).
