#!/bin/bash


# 检查是否以root权限运行
if [[ $EUID -ne 0 ]]; then
    echo "错误：本脚本必须使用 root 权限运行" >&2
    exit 1
fi

# 检查用户列表文件是否存在
if [ ! -f "user_list.txt" ]; then
    echo "错误：user_list.txt 文件未找到" >&2
    exit 1
fi

# 逐行处理用户列表
while IFS= read -r username; do
    # 过滤空行和空白字符
    username_clean=$(echo "$username" | tr -d '[:space:]')
    if [ -z "$username_clean" ]; then
        continue
    fi

    # 检查用户是否存在
    if id "$username_clean" &>/dev/null; then
        continue  # 用户已存在则跳过
    else
        # 创建用户并检查结果
        if useradd -m "$username_clean" &>/dev/null; then
            echo "用户 <$username_clean> 创建成功！"
        else
            echo "错误：用户 <$username_clean> 创建失败" >&2
        fi
    fi
done < "user_list.txt"

echo "用户创建流程完成"
