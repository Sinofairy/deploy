# 模型编译

## 使用方法
```shell
python3 upload_compile_model.py --pub_config cfg/pub_cfg.py --compile-mode local/aidi
```

## 运行流程
1. int_infer模型生成（本地）：（`check_and_convert`方法）
   1. 根据`cfg/pub_cfg.py`里每个模型`update_cfg`字段所声明的阈值，更新config里的desc信息。
   2. 根据模型checkpoint的参数信息，以及对应的模型cfg中所定义的模型结构，通过trace的方式得到int_infer模型。

2. 编译得到hbm上板模型
   1. 本地（local）模式 （`compile_local`方法）：轮询`pub_cfg`中定义的各模型设置，并按照读取得到的相应参数，编译对应的int_infer模型。最后调用`hbdk-pack`命令将编译好的所有模型打包。
   2. 艾迪（aidi）模式
      1. `upload_to_aidiexp`方法：轮询`pub_cfg`中定义的各模型设置，将上一步trace得到的int_infer模型通过`aidisdk`上传至艾迪实验模型管理平台中，并记录相关信息。
      2. `compile_publish_aidi`方法：轮询`pub_cfg`中定义的各模型设置，对每个待编译模型，输入其对应实验模型信息，并按照`pub_cfg`中的相应内容设置参数。最后提交编译任务，等待编译完成。
