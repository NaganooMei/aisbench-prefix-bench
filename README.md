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
前缀（长度 = input_len × hit_rate，按 block_tokens 向下取整）
  + N 个唯一随机 token（默认 3）
  + 随机后缀（补足到 input_len）
```

默认 **一条请求一条独立前缀**（不配置 `prefix_num` 时等于 `num_requests`）。需要模拟多请求共享同一前缀时显式配置 `prefix_num`，请求按 `i % prefix_num` 分组共用前缀。

因为前缀之后的 token 每条都不同，只有前缀能被复用，实际可达命中率就等于 `hit_rate`。`seed` 控制唯一 token 与后缀。

## 预埋（seed）与预热（warmup）是两件事

| | 做什么 | 条数 | 目的 |
| --- | --- | --- | --- |
| **预埋** | 只发前缀本身，把每个前缀写进缓存 | 前缀组数 × DP 数 | 让正式请求能命中 |
| **预热** | 用 4 条 8K 请求跑一遍完整循环（预埋 → 清缓存 → 加载） | 每组 4 条 × DP 数 | 把 **load 流程**（图捕获、加载线程、缓冲区）跑热，避免正式 case 吸收首轮开销 |

要点：

- 预埋和预热都**按 DP 定向**：通过 `X-data-parallel-rank` 头把请求钉到指定 rank，每个 DP 都收到完整的前缀集合，不靠内部 LB 碰运气。
- 预热在每个 mode 开始前各做一次（HBM 不 reset，cache 清 HBM，posix-reset 清 HBM+UCM），因为三种模式的 load 路径不同。
- 预热数据使用独立 seed，不进汇总表；参数可在 `warmup` 里覆盖（默认 8K 输入、4 请求、4 并发、90% 命中、输出 1 token）。

## 三种模式

| 模式 | 预埋 | 清缓存 | 说明 |
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
    {"name": "svc-a", "base_url": "http://127.0.0.1:8000",
     "served_model": "model-a", "model_path": "/models/model-a", "dp": 1}
  ],
  "cases": [
    {"input_len": 8192, "output_len": 1, "hit_rate": 0.9,
     "concurrency": 8, "num_requests": 16, "request_rate": 0}
  ],
  "modes": ["hbm", "cache", "posix-reset"],
  "block_tokens": 128,
  "corpus_path": "assets/corpus.txt"
}
```

关键字段：

- `services[]`：**一条配置可以放多个服务**，同一组 case 会在每个服务上各跑一遍，最终进同一张表。
- `cases[]`：`hit_rate` 必须在 (0,1) 开区间；`concurrency <= num_requests`；`request_rate=0` 表示不限速。`prefix_num` 可选，不填即"一请求一前缀"，填了就是若干个请求共享同一条前缀。
- `services[].dp`：数据并行的 DP 域数量，决定预热和预埋要在几个 rank 上各做一遍。
- `warmup`：预热组参数，默认 `{"input_len": 8192, "num_requests": 4, "concurrency": 4, "hit_rate": 0.9, "output_len": 1}`，`enabled=false` 可关闭。
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
Service      Mode         Input  Target  HBM hit  Ext hit  Hit tok  Prefill tok  TTFT ms  ...
-----------  -----------  -----  ------  -------  -------  -------  -----------  -------
svc-a        hbm          8K     89%     86.9%    0.0%     114,688  17,217       1,521.9  ...
svc-a        cache        8K     89%     0.0%     97.8%    129,024  2,881        674.7    ...
svc-a        posix-reset  8K     89%     0.0%     97.8%    129,024  2,881        700.8    ...
```

- `Target`：按 `block_tokens` 对齐后实际可达的目标命中率（例如配置 0.9、8K 输入时是 7296/8192 = 89%）。
- `HBM hit`：`vllm:prefix_cache_hits / queries` 增量，分母是查询 token。
- `Ext hit`：`vllm:external_prefix_cache_hits / queries` 增量，即 UCM/SGLang HiCache 等外部缓存贡献的命中；**分母是扣除 HBM 命中后剩余的 token**，所以它和 `HBM hit` 不能直接相加。
- `Hit tok` / `Prefill tok`：跨模式对比只看这两列——命中 token 绝对量和仍然需要重算的 token 数。
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
5. `ImportError: pyarrow requires NumPy 2.0 or newer` —— `ais-bench-benchmark 3.1.20260630` 的元数据把 numpy 钉在 `<2.0`，但它依赖的 pyarrow 需要 `>=2.0`。在与 vLLM 隔离的 venv 里直接 `pip install "numpy>=2"` 即可，实测 ais_bench 运行正常；不建议装在 vLLM 主环境。
6. TPOT 一直是 `n/a` —— `output_len=1` 没有 decode 阶段，没有 TPOT 样本；要测 TPOT 请把 `output_len` 调大。

## 本地检查

```bash
python -m unittest discover -s tests -v
```

测试不依赖 AISBench、NPU 或网络，覆盖配置校验、数据集形状、命中率计算和出表格式。
