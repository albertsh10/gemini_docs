Project Structure
=================

| gemini
| ├── bin
| │   ├── gemini_python.py
| │   ├── autotuner.py
| │   └── __init__.py
| ├── code_tree
| │   ├── code_node_base.py
| │   ├── code_node_leaf.py
| │   ├── code_node_root.py
| │   └── __init__.py
| ├── gemini_compiler.py
| ├── __init__.py
| ├── pass_manager
| │   ├── import_module_pass_manager.py
| │   ├── __init__.py
| │   ├── passes
| │   │   ├── import_module_pass.py
| │   │   ├── __init__.py
| │   │   ├── matmul_sharding_pass.py
| │   │   ├── pass_base.py
| │   │   ├── pipeline_device_stage_pass.py
| │   │   ├── plugin_dense_pass.py
| │   │   ├── plugin_dropout_pass.py
| │   │   ├── plugin_gather_pass.py
| │   │   ├── plugin_import_fix_pass.py
| │   │   ├── plugin_layer_norm_pass.py
| │   │   ├── plugin_matmul_pass.py
| │   │   ├── plugin_multiply_pass.py
| │   │   ├── plugin_reshape_pass.py
| │   │   ├── plugin_softmax_pass.py
| │   │   └── plugin_transpose_pass.py
| │   ├── pass_manager_base.py
| │   ├── pass_registry.py
| │   ├── pipeline_pass_manager.py
| │   ├── sharding_pass_manager.py
| │   └── transformer
| │       ├── bert_device_placer_transformer.py
| │       ├── import_module_transformer.py
| │       ├── __init__.py
| │       ├── matmul_sharding_operand_transformer.py
| │       ├── matmul_sharding_operation_transformer.py
| │       ├── node_transformer_base.py
| │       ├── node_visitor_base.py
| │       ├── plugin_dense_transformer.py
| │       ├── plugin_dropout_transformer.py
| │       ├── plugin_gather_transformer.py
| │       ├── plugin_import_fix_transformer.py
| │       ├── plugin_layer_norm_transformer.py
| │       ├── plugin_matmul_transformer.py
| │       ├── plugin_multiply_transformer.py
| │       ├── plugin_reshape_transformer.py
| │       ├── plugin_softmax_transformer.py
| │       └── plugin_transpose_transformer.py
| ├── plugins
| │   ├── api_wrapper.py
| │   ├── bert_plugin.py
| │   ├── __init__.py
| │   └── monad.py
| └── utils
|     ├── class_utils.py
|     ├── configuration.py
|     ├── file_util.py
|     ├── graph_util.py
|     ├── import_util.py
|     ├── __init__.py
|     └── logging_util.py

In the folder structure above:

- ``gemini/bin`` is the folder we implement our executable entries. Including gemini_python and autotuner 
- ``gemini/code_tree`` is the tree data structure that maintains the source code and its compiled AST. The tree structure follows the import dependency of the model codes, where the root node refers to the entry python file of the AI model.
- ``gemini/gemini_compiler`` is the implementation of the main compiler class that handles the whole gemini procedures on python AST:

  * parse -->
  * import -->
  * compile to AST -->
  * apply passes -->
  * execute on interpreter.

- ``gemini/pass_manager`` is a pass module that implements sorts of pass managers that used for applying model parallel patterns to the python AST. In general, it consists of `base classes` of pass_manager / pass_registry / passes / and transformers.

  * Pass Registry is a global registry that holds the <id, pass class> pair globalwise
  * Pass Managers and its base class, managers a bunch of passes that work for a common purpose, such as `sharding mode`. it exposes ``llvm``-like interfaces, such as ``add_pass``, ``schedule_pass``, ``run_pass``.
  * Passes and its base class managers a bunch of specific transformers, that inherit from ``python::ast::NodeVisitor`` and ``python::ast::NodeTransformer``.
  * transformers, do transformations and visiting on python AST, do works on recognised DSL nodes.

- ``gemini/plugins`` provides a convinient way to implement complex python AST nodes for injection into the target AST. this provides fruitful functionalities that enables you to write advanced python features dynamically by passes.
- ``gemini/utils`` is the utility module, that provides basic supports on commonly used functions / methods
