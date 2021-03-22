Play with Bert-large
====================

You can follow these steps to run the bert-large with gemini.

Download the datasets
---------------------

download squad dataset:

.. code-block:: shell

    wget ftp://10.16.11.32/software/dataset/squad.zip

download glue dataset:

.. code-block:: shell

    wget ftp://10.16.11.32/software/dataset/glue.zip

Download the pretrained checkpoints:
For Bert Base:

.. code-block:: shell

    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    
Verify if the download succeed by:

.. code-block:: shell

    get_sha256sum() {
      cat $1 | sha256sum | head -c 64
    }
    
    string_contains () {
      [ -z "$1"  ] || { [ -z "${2##*$1*}"  ] && [ -n "$2"  ]; };
    }

    if string_contains "acae5418a2f9c301fc5ac327e75de05d1bd57c5f17667faa270555318963526c" `get_sha256sum uncased_L-12_H-768_A-12.zip`

For Bert Large:

.. code-block:: shell

    wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip

Verify if the download succeed by:

.. code-block:: shell

    get_sha256sum() {
      cat $1 | sha256sum | head -c 64
    }
    
    string_contains () {
      [ -z "$1"  ] || { [ -z "${2##*$1*}"  ] && [ -n "$2"  ]; };
    }

    if string_contains "3619df03637b027f41f4bf6d3ea21d4f162129dd57b6cc223939a320046a1ec5" `wwm_uncased_L-24_H-1024_A-16.zip`

Unzip the dataset and pretrained checkpoints
--------------------------------------------
unzip the dataset under folder ${BERT_DATASET}
unzip the checkpoints under folder ${BERT_CKPT}

Run Squad pretrain
------------------
assumes the root path of bert model codes are ``${BERT_DIR}``

.. code-block:: shell

    export SQUAD_DIR=${BERT_DATASET}/squad/v1.1
    export OUT_DIR=${BERT_DIR}/squad_output
    nohup gemini_python ${BERT_DIR}/run_squad.pay \
      --vocab_file=${BERT_CKPT}/vocab.txt \
      --bert_config_file=${BERT_CKPT}/bert_config.json \
      --init_checkpoint=${BERT_CKPT}/bert_model.ckpt \
      --do_train=True \
      --do_predict=True \
      --device=dtu \
      --train_file=$SQUAD_DIR/train-v1.1.json \
      --predict_file=$SQUAD_DIR/dev-v1.1.json \
      --train_batch_size=1 \
      --learning_rate=5e-6 \
      --num_train_epochs=0.003 \
      --max_seq_length=128 \
      --doc_stride=128 \
      --output_dir=$OUT_DIR \
      --use_resource=False \
      --use_xla=True \
      --horovod=False \
      --display_loss_steps=10

if you want to enhance the performance by mute the one master-piece weights that are compatible with pretrained checkpoints but not fast. just comment the arguments on ``--init_checkpoint``.

Run MRPC pretrain
------------------
assumes the root path of bert model codes are ``${BERT_DIR}``

.. code-block:: shell

    export SQUAD_DIR=${BERT_DATASET}/squad/v1.1
    export OUT_DIR=${BERT_DIR}/squad_output
    nohup gemini_python ${BERT_DIR}/run_classfier.py \
      --task_name=MRPC \
      --vocab_file=${BERT_CKPT}/vocab.txt \
      --bert_config_file=${BERT_CKPT}/bert_config.json \
      --init_checkpoint=${BERT_CKPT}/bert_model.ckpt \
      --do_train=true \
      --do_eval=false \
      --device=dtu \
      --data_dir=${GLUE_DIR}\
      --max_seq_length=128 \
      --train_batch_size=1 \
      --learning_rate=2e-5 \
      --num_train_epochs=0.01 \
      --use_resource=False \
      --use_xla=True \
      --horovod=False \
      --output_dir=${OUT_DIR}

if you want to enhance the performance by mute the one master-piece weights that are compatible with pretrained checkpoints but not fast. just comment the arguments on ``--init_checkpoint``.