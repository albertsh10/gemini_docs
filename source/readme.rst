What is Gemini?
===============
Gemini will solve your problem, if you wish to run your AI model in a manner of model parallel without any pain and effort in modifying/transplanting/suiting your model code.

GEMINI is the abbreviation of Generic Model Parallel Instrumenter. It generically suppport any main-stream AI frameworks, and make supports on its domain-specific data-flow language.

How Gemini Works?
-----------------
The essences of Gemini make plays on the python ASTs, and recognise any supported types of DSL that used by AI framework to construct models. e.g. ``tensorflow.nn.dropout`` in ``tensorflow``, and ``torch.nn.conv2d`` in ``torch``
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

- Be awesome
- Make things faster

Installation
------------

Install $project by running:

    install project

Contribute
----------

- Issue Tracker: github.com/$project/$project/issues
- Source Code: github.com/$project/$project

Support
-------

If you are having issues, please let us know.
We have a mailing list located at: project@google-groups.com

License
-------

The project is licensed under the MIT license.