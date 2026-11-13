# aisbench-prefix-bench

使用 AISBench 执行与 [vllmbench](https://github.com/NaganooMei/vllmbench/tree/4801ebb) 相同的缓存性能测试流程，保留多服务配置。数据构造沿用 [aisbench_auto_tools_prefix](https://github.com/rayn-zzz/aisbench_auto_tools_prefix) 的语料方式。

## 快速使用

在系统已有的 AISBench Python 环境运行，确保该环境可以导入 `ais_bench` 和 `transformers`，并且 AISBench 的配置目录可写。先复制 [bench.example.json](bench.example.json)，填写实际服务地址、模型路径、DP 和 `corpus_path`。

```bash
cp bench.example.json bench.json
python bench.py cache bench.json --dry-run
python bench.py cache bench.json
python bench.py hbm-shared-v4 bench.json
```

`corpus_path` 指向已有的 GSM8K JSONL 或纯文本文件。程序从这些问题行构造前缀，`work_dir/picked_ids_*.json` 记录已使用的语料行。一次命令可以执行一个模式中的全部服务和 case；`run` 入口按配置中的 `modes` 执行矩阵。

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

## 配置示例

下面使用 32K/40 并发和 128K/10 并发两个 case，输出均为 512 token。服务地址、模型和语料路径需要填写现场值。

```json
{
  "services": [
    {
      "name": "svc-a",
      "base_url": "http://localhost:18081",
      "served_model": "glm",
      "model_path": "/models/xxx",
      "dp": 4
    }
  ],
  "cases": [
    {"input_len": 32768, "output_len": 512, "hit_rate": 0.9, "concurrency": 40, "num_requests": 40},
    {"input_len": 131072, "output_len": 512, "hit_rate": 0.9, "concurrency": 10, "num_requests": 10}
  ],
  "modes": ["hbm", "cache", "posix-reset"],
  "corpus_path": "/data/datasets/GSM8K.jsonl",
  "work_dir": "work",
  "results_dir": "results",
  "seed": "random",
  "ucm_warmup": {
    "enabled": true, "rounds": 1,
    "input_len": 8192, "output_len": 1,
    "hit_rate": 0.9, "concurrency": 4, "num_requests": 4
  },
  "metrics": {"enabled": false}
}
```

| 字段 | 含义与默认值 |
| --- | --- |
| `services[].dp` | 服务的 DP 域数量 |
| `services[].dp_rank` | 可选固定 DP，CLI 的 `--dp-rank` 可覆盖 |
| `input_len`、`output_len` | 每条正式请求的输入和输出 token 数 |
| `concurrency`、`num_requests` | 整个正式组的并发和请求数，要求并发不超过请求数 |
| `hit_rate` | 目标比例，范围 0～1；单独执行 `hbm-no-cache` 时可省略 |
| `block_tokens` | 可选预埋长度对齐参数，默认 1；两个 V4 模式固定按 4096/32 计算 |
| `ucm_warmup` | 默认一轮 8K/4 请求/4 并发的独立预热，也兼容 `warmup` 字段 |
| `seed_settle_seconds` | 预埋后的等待时间，默认 1 秒 |
| `metrics.enabled` | 详细采集开关，默认 false；正式窗口的轻量计数默认读取 |
| `aisbench.command` | AISBench 可执行文件，默认 `ais_bench` |
| `aisbench.configs_dir` | 可选配置目录，默认从当前 Python 环境的 AISBench 安装位置定位 |

普通模式的目标预埋长度按 `block_tokens` 向下取整，100% 目标会预留尾部重算空间。V4 模式会为额外 32 token 保留空间，32768、100% 对应的预埋长度也为 28704。取整为 0 时跳过该组预埋。目标命中率与对齐后的预埋长度分别记录在结果中。

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
- Prefill 吞吐请查看 AISBench 原生性能报告中的 `PrefillTokenThroughput`，具体位置见下文。
- HBM hit 与 External hit 都除以 HBM queried tokens。`external_hit_rate_remaining` 另存扣除 HBM 后的外部命中口径。
- 默认读取正式组前后的 `/metrics` 计算命中率，缺指标显示 `n/a`。`--collect-metrics` 额外导出原始快照、按 rank 的 CSV/Excel、直方图和耗时统计，采集失败按错误处理。

指标采集和 Excel/吞吐代码复用 vllmbench `4801ebb`，保存在 `abpb/collect_ucm_metrics.py`、`metrics_capture.py`、`vllm_reporting.py`。AISBench 版适配了输入结果和 Service 列，Prefill 吞吐交由 AISBench 原生报告呈现。

原始 AISBench 产物及数据放在 `work_dir`。结果、manifest 和汇总放在 `results_dir`。旧的 `case_result.json` 仍可用 `report` 出表。

### 汇总列口径

| 列 | 计算口径 |
| --- | --- |
| `Output tok/s` | AISBench 正式轮次报告的输出 token 吞吐 |
| `Total tok/s` | AISBench 正式轮次报告的输入加输出 token 吞吐 |
| `Input tok/s` | 总吞吐减输出吞吐 |
| `HBM hit` | HBM hit token 增量 ÷ HBM query token 增量 |
| `External hit` | external hit token 增量 ÷ HBM query token 增量 |
| `Host Ready` | 加载时已 Ready 的 Host Cache shard 数 ÷ 成功加载 shard 数 |
| `Posix bytes` | `ucm:posix_s2h_bytes_total` 在正式窗口的增量，汇总各 worker，单位 bytes |

输入吞吐包含命中缓存的 token。External hit 包含 Host Cache 和 Posix 的命中贡献，加载来源可结合 Host Ready、backend load 和 Posix bytes 判断。计数器真实零增量显示 `0`，缺失的指标显示 `n/a`。

### AISBench 原生 Prefill 指标

在 `work/<run>/<service>/<mode>/<case>/measure/` 下的 AISBench 原生性能报告，以及相邻的 `measure.log` 中查看 `PrefillTokenThroughput`。当前 AISBench 主线逐请求计算输入 token 数除以 TTFT，再对这组值计算 Average、Median 和分位数。较早版本使用输入 token 总数除以 TTFT 总和，具体展示以安装版本为准。官方的口径变更记录见 [PR #427](https://github.com/AISBench/benchmark/pull/427/files)。

### 产物与输出长度

| 路径或阶段 | 内容 |
| --- | --- |
| `work/<run>/<service>/<mode>/<case>/prefix/`、`full/` | 保存预埋和完整请求的数据集 |
| 同级 `seed_auto/` 或 `seed_dp<rank>/` | 预埋的 AISBench 产物，输出固定 1 token |
| `warmup_*/warmup_load_dp<rank>/` | 加载预热产物，默认输出 1 token |
| 正式 case 的 `measure/` 和 `measure.log` | 完整正式请求的原生结果与日志，输出使用 case 配置 |
| `results/<run>/<service>/<mode>/manifest.json` | case、seed、预热和 Posix 数据复用记录 |
| `results/<run>/<service>/<mode>/<case>/case_result.json` | 正式轮次的参数、性能、命中率和错误 |
| `results/summary.csv`、`summary.xlsx` | 当前结果目录的汇总 |

Posix 重启后的原始数据路径由预埋 manifest 保存。保留对应的 `work_dir` 数据文件，`posix-run` 可按记录复用。预埋输出 1 token、正式输出 128 token 是两个阶段各自的配置；正式请求的实际输入、输出 token 总数会受到校验。

```bash
# 重新生成终端表格、CSV 和 Excel
python bench.py report results
```

## 本地验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q abpb bench.py
```

测试使用模拟 AISBench 和 tokenizer，覆盖九种模式的路由、并发、预热、长度边界、Posix 重启复用、失败处理及配置隔离。真实 AISBench 与 NPU 性能需要在目标环境验证。
