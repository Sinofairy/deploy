编译docker环境:reg-ai.chehejia.com/j5/hat_pilot_development_toolkit:1.1.1 
启动docker，把hbdk版本升到3.28.2，plugin版本不动，multitask.py里march改成bayes

带车灯的模型编译, 文件替换说明:
classification.py 替换hat/core/proj_spec/classification.py
Detection.py 替换hat/core/proj_spec/detection.py
pub_cfg_x3c_mwl.py 中有关于rear_ligth阈值的设置
训练网络配置之前的压缩包里有，pilot_rear_light/configs/resize_2/multitask.py

编译命令:python3 -u projects/pilot/model_release/upload_compile_model.py --pub-config projects/pilot/model_release/cfg/pub_cfg_x3c_20220608.py --compile-mode local

编译工具的各历史版本在: compile_softwares文件夹