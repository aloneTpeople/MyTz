#!/bin/bash

# 定义输出文件
output_file="system_report.txt"

# 清空或创建输出文件
> "$output_file"

# 1. 显示当前用户
echo "1. 当前用户:" >> "$output_file"
whoami >> "$output_file"
echo "" >> "$output_file"

# 2. 显示当前系统时间
echo "2. 当前系统时间:" >> "$output_file"
date >> "$output_file"
echo "" >> "$output_file"

# 3. 显示CPU负载情况
echo "3. CPU负载情况:" >> "$output_file"
uptime >> "$output_file"
echo "" >> "$output_file"

# 4. 显示磁盘使用情况
echo "4. 磁盘使用情况:" >> "$output_file"
df -h >> "$output_file"
echo "" >> "$output_file"

# 5. 显示当前内存使用情况
echo "5. 内存使用情况:" >> "$output_file"
free -m >> "$output_file"
echo "" >> "$output_file"

echo "系统信息已收集并保存到 $output_file"
