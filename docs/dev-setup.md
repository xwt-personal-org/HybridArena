# 本地开发说明

## Python 版本

项目支持 `3.10 <= version < 3.13`。建议使用 Python 3.10、3.11 或 3.12。

## 安装

```bash
pip install -e ".[dev,rl]"
```

## 基础验证

```bash
python -m compileall hybrid_arena
pytest hybrid_arena/minimoba/tests -v
ruff check hybrid_arena
python hybrid_arena/scripts/benchmark_fps.py
```
