# aisbench-prefix-bench

使用 AISBench 执行与 [vllmbench](https://github.com/NaganooMei/vllmbench/tree/4801ebb) 相同的缓存性能测试流程，保留多服务配置。数据构造沿用 [aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) 的语料方式。

## 模式与预埋

| 模式 | 预埋请求与 DP 路由 | 正式测量前操作 |
| --- | --- | --- |
| `hbm` | 每个 DP 预埋完整请求集合，逐 DP 执行 | 保留 HBM |
| `v4hbm` | 每个 DP 预埋完整请求集合，逐 DP 执行，使用 V4 长度规则 | 保留 HBM |
| `hbm-shared` | 一条公共前缀，各 DP 并行预埋，每个 DP 1 条、并发 1 | 保留 HBM |
| `hbm-shared-v4` | 一条公共前缀，逐 DP 预埋，每个 DP 1 条、并发 1，使用 V4 长度规则 | 保留 HBM |
| `hbm-no-cache` | 直接发送完整正式请求 | 只执行正式组 |
| `cache` | 每个 DP 预埋完整请求集合，各 DP 并行执行 | `reset_external=false` |
| `posix-seed` | 完整集合发送一次，默认由内部 LB 分配 | 保存预埋记录，等待写出后由用户停服重启 |
| `posix-run` | 读取已保存的数据和 seed | 重启后执行加载预热与正式请求 |
| `posix-reset` | 完整集合发送一次，默认由内部 LB 分配 | `reset_external=true`，清 HBM 和 UCM Cache |

完整集合按 case 中的 `num_requests` 和 `concurrency` 执行。例如 40 请求、40 并发、DP4，`cache` 同时向每个 DP 各发送 40 条，Posix 总共只发送 40 条。定向请求携带 `X-data-parallel-rank` header。可通过 `--dp-rank` 或 `services[].dp_rank` 将流程固定到指定 DP。正式组默认使用内部 LB。

V4 模式先将 `input_len × hit_rate` 按 4096 向下取整，再加 32 token 作为预埋长度。32768、90% 对应预埋 28704，预期 HBM token 为 28672。规则固定为 4096/32，100% 时保留尾块空间，取整为 0 时跳过预埋。

`hbm-shared` 的预埋输入为公共前缀加 1 token。`hbm-shared-v4` 的公共前缀包含额外 32 token，预埋总输入正好是上述 V4 预埋长度。每个 case 的正式总输入长度始终使用配置值。

## 数据和 seed

前缀从未使用的 GSM8K 问题行中挑选，重复或截取到目标长度。完整请求由语料前缀、最多 3 个随机 token 和语料后缀组成。模式决定前缀组数，普通模式每条请求独立，shared 模式共用一条。

通过 `corpus_path` 指定 GSM8K JSONL 或纯文本。语料需要有足够的不同问题行来覆盖全部 case、预热及重复实验。仓库的 `assets/corpus.txt` 是小规模格式示例，正式测试可使用参考项目的 [GSM8K.jsonl](https://github.com/rayn-zzz/aisbench_auto_tools_prefix/blob/main/GSM8K.jsonl)。

与参考项目的 `picked_ids.txt` 一样，已选语料行保存在 `work_dir/picked_ids_*.json` 中，跨 mode 和跨执行轮换前缀。清理服务缓存后可以清理这份选取记录，或使用新的 `work_dir`。不同 seed 控制随机 token 和语料选择，但固定 seed 的结果也受已选记录影响。Posix 重启测试直接复用保存的数据文件和 seed。

默认每个 case、mode 和预热组各生成一个随机 32 位 seed。可以用根配置或 case 的整数 `seed` 复现随机序列，或写 `"random"`。一次预埋与正式测量复用同一份数据，程序会检查完整输入长度，以及预埋前缀与正式请求在 token 层面的对应关系。

请求使用 AISBench 的流式 `/v1/completions` 模型，传输纯文本 prompt。请求设置 `ignore_eos=true` 和 `add_special_tokens=false`。每阶段检查请求成功数、失败数、实际输入 token 总数及输出 token 总数，异常会停止本模式并保存错误。

## 使用

运行环境需要已有 AISBench、transformers 和模型 tokenizer。控制程序与报表使用 Python 3.10+ 标准库。修改 `bench.example.json` 的服务、模型、语料路径和 case 后执行。

```bash
python bench.py hbm bench.example.json
python bench.py hbm-shared-v4 bench.example.json
python bench.py hbm-no-cache bench.example.json
python bench.py cache bench.example.json
python bench.py posix-reset bench.example.json
```

```bash
python bench.py posix-seed bench.example.json
# 等待 UCM 写出完成，停服重启，保留 Posix 文件和 KV key/分片配置
python bench.py posix-run bench.example.json
# 也可以显式指定预埋 manifest 或其目录
python bench.py posix-run bench.example.json --run results/<run>/<service>/posix-seed
```

各 DP 需要访问同一套 Posix 文件，KV key/分片配置要兼容。预埋后等待时间为 `seed_settle_seconds` 与 `metrics.settle_seconds` 的较大值，默认 1 秒。Posix 写出完成仍需结合服务端状态确认。

```bash
python bench.py cache bench.example.json --dp-rank 0
python bench.py cache bench.example.json --collect-metrics
python bench.py run bench.example.json --dry-run
python bench.py report results
```

原来的 `python bench.py run <config>` 仍按 `modes` 顺序执行多服务、多 case 矩阵，默认 modes 为 `hbm`、`cache`、`posix-reset`。重启 Posix 的两个阶段需要分别运行。每个 AISBench 调用使用独立临时配置名称，支持各 DP 同时启动。

## 预热与清缓存

默认各模式执行一轮独立 8K 预热，4 请求、4 并发、90% 前缀、输出 1 token。配置支持 `ucm_warmup`，也兼容旧的 `warmup`，可设置 `enabled`、`rounds` 和 case 参数。预热数据按当前模式的预埋规则处理，命中阶段覆盖各目标 DP。V4 预热也采用 4096 向下取整加 32 规则。

`hbm-no-cache` 直接执行正式组。裸推基线使用关闭 prefix caching、停用 UCM 的服务配置。

reset 严格检查布尔返回值。空 HTTP 200、非法响应或超时会结束本模式。任何预埋或预热失败也会结束本模式。`posix-reset` 清空 HBM 和 UCM 内存缓存，保留 Posix 文件。计数窗口从 reset 成功后开始。

## 结果

每个 mode 完成后输出终端表格和 `summary.csv`、`summary.xlsx`，包含模型、月日时分、TTFT、TPOT、输入/输出/总吞吐、HBM/External hit、Host Ready 和 Posix bytes。

- 输入吞吐按总吞吐减输出吞吐计算。
- Prefill 吞吐使用 AISBench 原生性能报告中的 `PrefillTokenThroughput`。
- HBM hit 与 External hit 都除以 HBM queried tokens。`external_hit_rate_remaining` 另存扣除 HBM 后的外部命中口径。
- 默认读取正式组前后的 `/metrics` 计算命中率，缺指标显示 `n/a`。`--collect-metrics` 额外导出原始快照、按 rank 的 CSV/Excel、直方图和耗时统计，采集失败按错误处理。

指标采集和 Excel/吞吐代码来自 vllmbench `4801ebb`，保存在 `abpb/collect_ucm_metrics.py`、`metrics_capture.py`、`vllm_reporting.py`，适配包导入、AISBench 结果和 Service 列，运行时无需安装 vllmbench。

原始 AISBench 产物及数据放在 `work_dir`。结果、manifest 和汇总放在 `results_dir`。旧的 `case_result.json` 仍可用 `report` 出表。

## 本地验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q abpb bench.py
```

测试使用模拟 AISBench 和 tokenizer，覆盖九种模式的路由、并发、预热、长度边界、Posix 重启复用、失败处理及配置隔离。真实 AISBench 与 NPU 性能需要在目标环境验证。
