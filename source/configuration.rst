How to Config Model Parallel Patterns
=====================================

We support running model parallel with json config files.
In the config file, you can simply customize the model parallel settings before run.

We support following hyper-parameters.

- Hyper Parameters ``mode``
   - Allowed Values - Type: string
      - ``sharding``
      - ``pipeline``
      - ``hybrid``
   - Meanings 
      - ``sharding`` mode means shard model layers into devices; 
      - ``pipeline`` mode refers to that split model into sequential stages, and pipelines the input data batch by batch. 
      - ``hybrid`` means ``sharding`` + ``pipeline`` 

- Hyper Parameters ``sharding_size``
   - Allowed Values - Type: int
      - ``1``
      - ``2``
      - ``4``
      - ``8``
      - any integers that can divide the total device cnt, (not recommended) 
   - Meanings 
      - ``sharding_size`` means the count of replicas when compute large op, layer or a sequence of layers in parallel. Compared to data parallel, this mode perform duplicates on inputs but sharding into replicas on weights. For DTU, a trivial setting is ``sharding_size=4`` that allows you perform auto4c-like behaviour; 


- Hyper Parameters ``sharding_dim``
   - Allowed Values - Type: int
      - ``0``
      - ``-1``
   - Meanings 
      - ``sharding_dim`` means the weight dimension that sharding performs on. ``0`` refers to sharding on reduction dimentions. that will result in a partial sum as output, hence followed by an all_reduce when merged from forks; another setting refers to sharding on the output dimensions that will result in part of the output tensor in each shard, and then calls for an all_gather when merge from forks. The trade-off between the two is not the key point in this tutorial. If you are interested, please quote the authors.

- Hyper Parameters ``stage_cnt``
   - Allowed Values - Type: int
      - larger than or equal to ``1``
   - Meanings 
      - ``stage_cnt`` means the stages that we chop the models into. Output tensor of one stage will feed to the next stage as input tensors, and vice versa in the gradient computation part.

- Hyper Parameters ``accum_degree``
   - Allowed Values - Type: int
      - larger than or equal to ``1``
   - Meanings 
      - ``accum_degree`` means how many prefetched data batches are loaded before a gradient synchoronization and weight apply. This params works in pair with ``stage_cnt``, which performs like a pipeline that when device at stage ``ith`` computing the ``kth`` data batch. the ``k-1 th`` data batch has been feed to ``i+1 th`` device for computation. Roughly, they runs at parallel manner.

- Hyper Parameters ``device_mapping``
   - Allowed Values - Type: 2-d dict
      - outer dict has keys in form of ``stage_<index>``, where the max index of stage equals to ``stage_cnt``.
      - inner dict has keys in form of ``shard_<index>``, where the max index of shard equals to ``sharding_size``.
   - Meanings 
      - ``device_mapping`` sets the rules that how the fractions of the model allocated at specific devices.

Example of a config file
------------------------

.. code-block:: json

    {
      "mode": "sharding",
      "sharding_size": 4,
      "sharding_dim": -1,
      "stage_cnt": 8,
      "accum_degree": 1,
      "device_mapping": {
        "stage_0": {
          "shard_0": "/device:XLA_DTU:0",
          "shard_1": "/device:XLA_DTU:1",
          "shard_2": "/device:XLA_DTU:2",
          "shard_3": "/device:XLA_DTU:3"
        },
        "stage_1": {
          "shard_0": "/device:XLA_DTU:5",
          "shard_1": "/device:XLA_DTU:6",
          "shard_2": "/device:XLA_DTU:7",
          "shard_3": "/device:XLA_DTU:8"
        },
        "stage_2": {
          "shard_0": "/device:XLA_DTU:10",
          "shard_1": "/device:XLA_DTU:11",
          "shard_2": "/device:XLA_DTU:12",
          "shard_3": "/device:XLA_DTU:13"
        },
        "stage_3": {
          "shard_0": "/device:XLA_DTU:15",
          "shard_1": "/device:XLA_DTU:16",
          "shard_2": "/device:XLA_DTU:17",
          "shard_3": "/device:XLA_DTU:18"
        },
        "stage_4": {
          "shard_0": "/device:XLA_DTU:20",
          "shard_1": "/device:XLA_DTU:21",
          "shard_2": "/device:XLA_DTU:22",
          "shard_3": "/device:XLA_DTU:23"
        },
        "stage_5": {
          "shard_0": "/device:XLA_DTU:25",
          "shard_1": "/device:XLA_DTU:26",
          "shard_2": "/device:XLA_DTU:27",
          "shard_3": "/device:XLA_DTU:28"
        },
        "stage_6": {
          "shard_0": "/device:XLA_DTU:30",
          "shard_1": "/device:XLA_DTU:31",
          "shard_2": "/device:XLA_DTU:32",
          "shard_3": "/device:XLA_DTU:33"
        },
        "stage_7": {
          "shard_0": "/device:XLA_DTU:35",
          "shard_1": "/device:XLA_DTU:36",
          "shard_2": "/device:XLA_DTU:37",
          "shard_3": "/device:XLA_DTU:38"
        }
      }
    }
