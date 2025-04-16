#!/bin/bash

# 定义要遍历的x值数组
# x_values=(4 5 6 7 8 9 10 11 12 15 20 25 30 35 40 45 50 55 60 65 70 75 80 90 100 120 140)
# x_values=(110 130 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300)
x_values=(800 900 1000)
# x_values=(2 3)

# 遍历数组中的每个x值
for x in "${x_values[@]}"
do
    echo "正在执行: python test.py --draft_token_number $x"
    python test.py --draft_token_number $x
    
    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
        echo "执行 python test.py --draft_token_number $x 时出错"
        exit 1
    fi
done

echo "所有任务执行完成"