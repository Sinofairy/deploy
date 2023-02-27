# Configs of models used in Pilot

## 环境准备

首先请参考[Installation Guide](../../docs/source/quick_start/installation.md)完成HAT基础开发环境的安装

随后，请在**本目录**执行

```shell
pip3 install horizon_plugin_pytorch_cu{cuda_version} -U -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc

pip3 install mxnet_horizon_cu{cuda_version} -U -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
```

以及

```shell
pip3 install --use-deprecated=legacy-resolver -r requirements.txt -i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc
```

来安装pilot相关模型训练所特别需要的专用依赖项。这里`cuda_version`可以是111或者102，取决于环境中torch对应的cuda版本。

## 运行

### 训练

#### 旧方法（已不适用于多任务模型）

首先将`GNUmakefile.template`复制到HAT根目录下，并改名为`GNUmakefile`.

随后即可使用其中预定义的命令来实现各种功能，如：

- `make train stage={stage}` （跑某个训练阶段的训练，默认为`with_bn`）
- `make train-pipeline stage={stage}` （跑某个训练阶段的pipeline测试， 默认为`with_bn`）
- `make submit` （提交训练job到AIDI集群），可以用`cluster=xxx`来指定运行job的集群

在各个命令中，都可以用`config={path_to_config}`来指定运行任务的入口config。其默认值为`projects/pilot/configs/multitask_dev/multitask.py`。

#### 新方法

请参考[相关文档](tools/train/README.md)中的方式启动多任务模型的训练。

### 集群化预测、评测

目前，pilot相关模型的预测评测需要使用`hpflow`包的`feature-hdlt`分支进行，请先从[这里](http://gitlab.hobot.cc/auto/perception/ad/hpflow/tree/feature-hdlt)得到代码。

接下来在当前目录运行`python3 tools/eval/submit_eval.py`命令即可，对应的参数有：

- `--dataset`: hpflow中评测集config的本地路径 
- `--eval-config`: hpflow中评测config的本地路径
- `--path-to-hpflow`: hpflow repo根目录的本地路径
- `--model-setting`: pilot模型的setting（一般是模组和场景的组合，如as33_day)
- `--model-config`: 可选参数，hat模型config的本地路径，如果不指定，则会在评测中使用`eval_config`中的默认值
- `--model-training-step`：可选参数，使用的模型对应的训练step，一般应与checkpoint对应，默认为`sparse_3d_freeze_bn_2`
- `--model-checkpoint`: 可选参数，hat模型config的本地路径，如果不指定，则会在评测中使用`eval_config`中的默认值。
- `--num-gpus`: 每个job用多少个gpu，默认为2
- `--num-worker-per-gpu`: 每个gpu起多少worker，一般来讲worker越多预测越快，也会占用更多的显存，默认为2
- `--cluster`: 跑预测任务的集群，目前推荐使用idc中的集群，否则会因为数据同步的问题导致预测很慢
- `--parallel`: 如果加入此参数，每个评测集会单独起一个job去跑。不加入的话，所有评测集会在一个job中执行，并且每跑完一个评测集提交一次评测任务
- `--report`: 如果加入此参数，会在预测任务结束后（如果是parallel模式，则是在最后一个评测集对应的预测任务之后）启动report任务，产生评测report。需要注意的是，由于最后一个评测集不一定最后跑完，因此report中可能存在评测集收集不全的情况，因此可能需要手动rerun。

其中`dataset`和`eval_config`为`hpflow`的相关config，具体配置可参考`hpflow`相关文档修改。

以上的所有路径中，除`--model_config`路径必须为相对于本目录(`projects/pilot`)的相对路径外，其他都可以是绝对路径或相对路径。