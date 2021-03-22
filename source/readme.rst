What is Gemini?
===============
Gemini will solve your problem, if you wish to run your AI model in a manner of model parallel without any pain and effort in modifying/transplanting/suiting your model code.

GEMINI is the abbreviation of Generic Model Parallel Instrumenter. It generically suppport any main-stream AI frameworks, and make supports on its domain-specific data-flow language.

How Gemini Works?
-----------------
The essences of Gemini make plays on the python ASTs, and recognise any supported types of DSL that used by AI framework to construct models. 
e.g. ``tensorflow.nn.dropout`` in ``tensorflow``, and ``torch.nn.conv2d`` in ``torch``.
A fruitful and extensible pass mechanism is design to apply transformations on the code AST, to enable sharding / pipeline and hybrid model parallel patterns on the code AST.
Look how easy it is to use. Imaging you have a model code with ``model_runner.py`` as your python entry

Defaultly, you will need to activate the environmental setting and then run with command:

.. code-block:: command 

    python model_runner.py *argv ...

When using Gemini, just do the following replacement on ``python`` command, magic happens:

.. code-block:: command 

    gemini_python model_runner.py *argv ...

Alternatively, you may wish to apply gemini on specific model function or a specified part of model code. Assume your model code looks like the follows:

.. code-block:: python 

    import tensorflow as tf

    def model(*args, **argv):
        # your model code snippets
        ...
        return logits, loss

    with tf.Session as sess:
        # your training logics
        ...

Now the magics for make ``model`` running on model parallel can be done by adding 2 lines of code:

.. code-block:: python 

    import tensorflow as tf
    # you need to import gemini library
    from gemini import GeminiCompiler 

    def model(*args, **argv):
        # your model code snippets
        ...
        return logits, loss

    # make compilation on your target model function 
    model = GeminiCompiler().compiler(model, config=GeminiConfig(mode=sharding_and_pipeline, sharding_size=4, pipeline_degree=8))

    with tf.Session as sess:
        # your training logics
        ...

Bingo! Now you can enjoy the boost by using model parallel in 


Features
--------

- For Users: Painless model parallel with your single device training code. You **Do not need to change any piece of code** to enable model parallel on main-stream AI frameworks, including Tensorflow and Pytorch.
- For Users: **Naturally compatible with different SW stacks with devices**. You do not worry about either the device is CPU/ GPU or DTU. just make sure the original code runs OK on single device.
- For Users: **Fruitful pattern of model parallel**. You can run with **sharding mode**, **pipeline mode** and in mix of the both.
- For Users: A **Clean Solution for large-scale training**. use horovod/mpi to enable data parallel on multiple server nodes. and Gemini will automatically chops the model into parts to fit in the computing devices within each server node.

- For Developers: A **Generic Pass Mechanism** that maximise the generality of model parallel functionality. The effort to support any new models is hence alleviated.
- For Developers: A **Easy to plugin interface** to implement new passes. Declarative-like method may support in future version. You will be able to implement passes with tbl-gen-like way.
- For Developers: **Round-trip functionality works along all the procedures**, that enables you debug model parallel passes without real run trials. **Saving time is saving money**.
- For Developers: **Graphviz support for python AST**, for debug use

Installation
------------

Step 1. Clone the code by command:

    git clone git@git.enflame.cn:heng.shi/gemini.git

Step 2. Initialize the project and install by:

    make init && make install

this command will install all the requirements, init the submodules and update them.

Entry of Gemini
---------------

If you want to try Gemini with samples, just type the Makefile entry `make samples/<sample_name>`

    make samples/

For instance, run **Bert-Large**, **mnist model**, and **python ast dump**, you can run with command:
    
    make samples/bert

    make samples/mnist

    make samples/dump_ast

If you want to test all the Gemini cases, type:

    make tests

If you want to clean up the code with PEP8 standard, do:

    make lint

    

Contribute
----------

- Gemini Project: git@git.enflame.cn:heng.shi/gemini.git
- Gemini documentation (also a submodule of Gemini project): git@github.com:albertsh10/gemini_docs.git 

Please make sure you have read through the code and understand the following aspects of the design thoughts:

- Model parallel basic concepts, some technical details will be a good plus.
- Compilation process of python
- Solid python fundamentals
- Meta-programming with python
- List Monad and other functional programming design patterns
- Handy experience with related toolchains, includes: cmake/Makefile, python ast module, functools, pep_linter, sphinx.

What's in coming next?
----------------------

- Autotuner
- Compatible solution with horovod (for multiple server node)
- Tuned Performance on GPU (on Gemini's own performance, regarding sharding and pipeline patterns)
- Tuned Performance on DTU (also including tuned OP and fusions on targeted models)
- 

Authors
-------

Albert Shi, Tianyu Jiang, Pilz Wang and Chris Liu

License
-------

The project is licensed under the MIT license.