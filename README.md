# aisbench-prefix-bench

在 AISBench 上做**前缀命中**测试的一键工具：造带前缀的数据集 → 预热 →（可选）清空缓存 → 用 AISBench 压测 → 直接出一张表。

与另外两个项目的分工：

| 项目 | 它解决的问题 | 本项目的差异 |
| --- | --- | --- |
| [AISBench](https://github.com/AISBench/benchmark) | 通用精度/性能评测 | 本工具只管前缀命中场景，并接管数据集与工作流 |
| [aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) | 前缀数据集生成 + 命中率统计 | 用配置文件代替改 `config.py`；增加清空缓存步骤；结构化解析结果而非正则扫屏 |
| [vllmbench](https://github.com/NaganooMei/vllmbench) | 基于 `vllm bench` 的分层命中对比 | 本项目以 AISBench 为压测器，适合需要 GSM8K 格式负载、精度复用或团队已有 AISBench 流水线的场景 |

## 数据集怎么构造

每条请求：

```
公共前缀（长度 = input_len × hit_rate，按 block_tokens 向下取整）
  + N 个唯一随机 token（默认 3）
  + 随机后缀（补足到 input_len）
```

因为前缀之后的 token 每条都不同，**只有公共前缀能被复用**，实际可达命中率就等于配置的 `hit_rate`；`prefix_num` 控制不同前缀的个数，`seed` 控制随机 token。

预热阶段只发送公共前缀本身（每 prefix、每 DP 各一次），所以正式请求能如预期命中，而不同 DP 的 engine 也都被覆盖。

## 三种模式

| 模式 | 预热 | 清缓存 | 说明 |
| --- | --- | --- | --- |
| `hbm` | 发前缀 | 不清 | vLLM 自身 HBM 前缀缓存的热态基线 |
| `cache` | 发前缀 | `reset_external=false` 只清 HBM | 保留 UCM 内存 Cache，测 Host Cache 命中 |
| `posix-reset` | 发前缀 | `reset_external=true` 清 HBM + UCM Cache | 保留 Posix 文件，测盘命中；不需要停服重启 |

reset 严格要求响应体是 `{"success": true}`（或裸 `true`）；空 200、非法 JSON、缺少布尔字段都会立即报错——否则会静默地测到未清空的缓存。

## 安装

需要目标环境里已有 AISBench（镜像通常自带；没有就 `pip install ais-bench-benchmark` 或源码安装），并且 `ais_bench` 命令可用、`benchmark/configs` 可写（工具会把两个临时配置文件放进去）。

```bash
git clone <this-repo>
cd aisbench-prefix-bench
python bench.py run bench.example.json --dry-run   # 只校验配置，不发请求
```

本工具自身只依赖 Python 3.10+ 标准库；`transformers` 用于按 token 精确构造数据集（AISBench 环境已有）。

## 配置

见 [bench.example.json](bench.example.json)。

```json
{
  "services": [
    {"name": "qwen38-27b", "base_url": "http://127.0.0.1:5261",
     "served_model": "Qwen3.8-27B", "model_path": "/mnt/model/Qwen3.8-27B", "dp": 1}
  ],
  "cases": [
    {"input_len": 8192, "output_len": 1, "hit_rate": 0.9, "prefix_num": 1,
     "concurrency": 8, "num_requests": 16, "request_rate": 0}
  ],
  "modes": ["hbm", "cache", "posix-reset"],
  "block_tokens": 128,
  "corpus_path": "assets/corpus.txt"
}
```

关键字段：

- `services[]`：**一条配置可以放多个服务**，同一组 case 会在每个服务上各跑一遍，最终进同一张表。
- `cases[]`：`hit_rate` 必须在 (0,1) 开区间；`concurrency <= num_requests`；`request_rate=0` 表示不限速。
- `block_tokens`：服务的实际缓存块长度。目标前缀按它向下取整，`8192 × 0.9 = 7372 → 7296`（128 对齐）。
- `corpus_path`：填充文本，支持 `{"question": ...}` 的 JSONL 或纯文本，可指向真实 GSM8K。
- `work_dir` / `results_dir`：原始产物与汇总表位置。
- `aisbench.configs_dir`：留空则自动定位；`aisbench.extra_args` 透传额外 CLI 参数。

## 运行

```bash
python bench.py run bench.example.json
```

输出：

```
Service      Mode          Input  Target  HBM hit  Ext hit  TTFT ms  TPOT ms  Out tok/s  Total tok/s  Load GB/s  Status
-----------  ------------  -----  ------  -------  -------  -------  -------  ---------  -----------  ---------  ------
qwen38-27b   hbm           8K     90%     89.1%    0.0%     ...
qwen38-27b   cache         8K     90%     0.0%     87.4%    ...
qwen38-27b   posix-reset   8K     90%     0.0%     86.9%    ...
```

- `Target`：按 `block_tokens` 对齐后实际可达的目标命中率（例如配置 0.9、8K 输入时是 7296/8192 = 89%）。
- `HBM hit`：`vllm:prefix_cache_hits / queries` 增量。
- `Ext hit`：`vllm:external_prefix_cache_hits / queries` 增量，即 UCM/SGLang HiCache 等外部缓存贡献的命中。
- `Load GB/s`：`ucm:cache_load_bytes_total` 与 `ucm:cache_load_duration_ms_sum` 折算的加载带宽，缺指标时显示 `n/a`。

每个格子还会落盘 `case_result.json`（配置、seed、reset 结果、性能与命中增量、错误），`results/summary.csv` 是汇总表。重新出表：

```bash
python bench.py report results/
```

## 判定原则

- **接口成功 ≠ 命中已验证**：reset 返回成功只代表清理动作被接受，必须同时看 `HBM hit` 和 `Ext hit` 是否符合预期。
- `cache`/`posix-reset` 的计数窗口从 reset 成功之后开始，预热不计入性能结果。
- `posix-reset` 清的是 UCM 内存 Cache，不清操作系统页缓存，因此和"停服重启"不是同一个冷态。

## 常见问题

1. `ais_bench is not importable` —— 当前 Python 环境没装 AISBench，或 `ais_bench` 命令与实际解释器不一致。
2. `Permission denied` 写 configs —— AISBench 装在了只读位置；用 `aisbench.configs_dir` 指向可写副本，或用 `pip install -e` 的源码环境。
3. 命中率远低于目标 —— 先确认 `block_tokens` 与服务一致，再确认预热是否覆盖所有 DP（看 `seed_rows` 日志），最后核对 `Ext hit` 的分母。
4. 旧版 AISBench 不认 `--num-warmups` —— 升级 AISBench；或把 `aisbench.extra_args` 里显式覆盖，但**不要**在没有 `--num-warmups 0` 的情况下测冷态，预热请求会污染结果。

## 本地检查

```bash
python -m unittest discover -s tests -v
```

测试不依赖 AISBench、NPU 或网络，覆盖配置校验、数据集形状、命中率计算和出表格式。
