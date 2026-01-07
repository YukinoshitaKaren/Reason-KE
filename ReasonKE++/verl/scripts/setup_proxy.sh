#!/bin/bash

# 设置 GitHub 代理的辅助脚本
# 使用方法: source setup_proxy.sh

echo "正在配置 GitHub 代理设置..."

# 设置 git 全局代理配置
git config --global url."https://ghfast.top/https://github.com/".insteadOf "https://github.com/"

# 设置环境变量（如果需要）
export GIT_TERMINAL_PROMPT=0

# 设置 pip 的信任主机（如果需要）
pip config set global.trusted-host "ghfast.top"

echo "代理配置完成！"
echo "现在所有对 github.com 的访问都会通过 ghfast.top 代理"

# 显示当前配置
echo "当前 git 配置："
git config --global --get-regexp url
