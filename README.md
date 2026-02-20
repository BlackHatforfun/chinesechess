# 思远象棋

基于 Python + Pygame Zero 的中国象棋项目，内置 AI 对弈，支持三档难度。

## 功能特性

- 三档难度：`简单`、`正常`、`大师`
- 支持人先手 / AI 先手
- 支持悔棋
- 将军红色横幅提示
- 中文棋子显示（含字体回退）

## 运行环境

- Python 3.10+
- `pgzero`
- `pygame-ce`

## 安装依赖

```bash
pip install pgzero pygame-ce
```

## 启动方式

```bash
python Chinese_Chess_Python.py
```

## 操作说明

- 鼠标：选择棋子并落子
- `U`：悔棋
- `A`：AI 先手新局
- `R`：人先手新局
- `E` / `N` / `M`：切换 `简单 / 正常 / 大师`

## 开源协议

MIT，见 `LICENSE`。
