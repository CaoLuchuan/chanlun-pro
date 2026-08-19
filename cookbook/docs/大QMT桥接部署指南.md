# 大 QMT 桥接（xtquant_big_convert）部署与改动说明

> 更新日期：2026-08-19
> 涉及仓库：[chanlun-pro](https://github.com/CaoLuchuan/chanlun-pro)（本仓库）、[xtquant_big_convert fork](https://github.com/CaoLuchuan/xtquant_big_convert)（上游 [litaolemo/xtquant_big_convert](https://github.com/litaolemo/xtquant_big_convert)）

---

## 1. 架构总览

chanlun-pro 通过大 QMT 桥接（ZMQ RPC）驱动券商的大 QMT 交易端，替代原 miniQMT（userdata_mini + xtquant SDK）方案：

```
chanlun-pro（Python 3.11, .venv）
    │  src/bigqmt_signal_trader        ← 桥接包（客户端，不入库，来自 fork 仓库）
    │  src/bigqmt_signal_trader_client_config.py  ← 客户端私有配置（资金账号，不入库）
    ▼
ZMQ RPC（端口按资金账号自动派生：15560 + 账号数字 % 100）
    │  A股 8886136661 → 15621    期货/期权 809222890 → 15650
    ▼
大 QMT 交易端内置 Python（服务端）
    │  D:\国金证券QMT交易端\python\BIGQMT_REDIS_DRYRUN.py   ← 入口（QMT 策略编辑器加载）
    │  D:\国金证券QMT交易端\python\bigqmt_signal_trader\     ← 桥接包（服务端副本）
    │  D:\国金证券QMT交易端\python\bigqmt_signal_trader_local_config.py  ← 服务端配置
    ▼
QMT 行情 / 交易（passorder、get_market_data_ex 等）
```

两条数据通路：
- **RPC 主路**：所有交易与行情请求，经 ZMQ ROUTER/DEALER
- **FormulaServer 直连快速路径**（只读行情，默认开启，连不上自动回退 RPC）

未部署桥接时自动回退 miniQMT（`exchange_qmt.py` 的 `USE_BIGQMT` 分支）。

---

## 2. 本次改动清单

### 2.1 chanlun-pro 侧（已推送）

| 文件 | 改动 |
|---|---|
| `pyproject.toml` | 新增 `pyzmq>=26.0.3`（zmq 传输必需，此前缺失导致 uv 同步时被卸载）；`requires-python` 限定 `>=3.11,<3.12`（ta-lib 本地 wheel 仅 cp311） |
| `src/chanlun/exchange/exchange_qmt.py` | 大 QMT 桥接模式（`USE_BIGQMT`）；无数据时触发服务端补历史下载后重试；K 线拉取 `timeout_seconds=120`（首次全量 1m 约 4 万根，实测需 115s，默认 6s 必超时） |
| `src/chanlun/trader/trader_qmt_stock/futures/option.py` | 交易端经兼容层 `XtQuantTrader(account_id=...)` ZMQ 连接 QMT；保留 miniQMT 回退分支 |
| `web/chanlun_chart/cl_app/__init__.py` | `/tv/history`：klines 为 None 返回 `no_data`（避免 500）；`/ticks`：RPC 超时返回空数据（避免 500） |
| `script/crontab/reboot_sync_qmt_{stock,futures}_klines.py` | 启动时自动切换到项目 `.venv` 重新执行（避免系统 Python 缺 pymysql/pyzmq）；支持 `CL_SYNC_LIMIT`/`CL_SYNC_WORKERS` 环境变量 |
| `.gitignore` | 排除 `src/bigqmt_signal_trader/`（桥接包）与 `src/bigqmt_signal_trader_client_config.py`（含资金账号） |

### 2.2 xtquant_big_convert fork 侧（已推送，基线 v0.2.3 + 1 commit）

fork 地址：https://github.com/CaoLuchuan/xtquant_big_convert
本地 clone：`d:\quantitative\xtquant_big_convert`（remote `origin` = fork，`upstream` = litaolemo）

| 文件 | 改动 |
|---|---|
| `transports/zmq_transport.py` | **服务端 handler 线程池**（`handler_workers`，默认 4，0 = 关闭回旧行为）：router loop 原先串行 inline 执行 handler，一次全历史拉取（分钟级）期间所有 tick/快请求排队至超时；现 loop 只收发，handler 分发线程池并行，工作线程响应经既有队列由 socket 归属线程回发（zmq socket 不可跨线程）。**客户端每请求独立 DEALER socket**：原共享 socket 在锁内等待整个超时窗口，一个 120s 拉取让并发短请求排队 2 分钟 |
| `xtquant_compat.py` | `_call` 把 `timeout_seconds` 混进 `params` 发给服务端、从未传给 `client.call` → 所有显式超时静默失效（一律 6s）；`download_history_data` 转发漏传 `incrementally` → 每次全量下载 |
| `tests/bigqmt_signal_trader/test_zmq_transport.py` | 新增 2 用例：慢 handler 不阻塞快请求、并发请求响应匹配。全套 374 passed |
| `CHANGELOG.md` | Unreleased 段落记录以上修复 |

上游 v0.2.3 已包含且我们同步部署的：#51 异步下单事件保序屏障、#52 财务表名映射、#54 下载日期窗口/单股下载兜底/本地缓存时间轴修复。

### 2.3 三处代码副本的同步关系

`src/bigqmt_signal_trader`（chanlun-pro，客户端）与 `D:\国金证券QMT交易端\python\bigqmt_signal_trader`（服务端）**已与 fork main 完全一致**（2026-08-19 验证，仅 `__pycache__` 差异）。

后续更新流程：
```powershell
# 1. fork 同步上游并推送
cd d:\quantitative\xtquant_big_convert
git fetch upstream; git merge --ff-only upstream/main; git push origin main
# 2. 同步到 chanlun-pro 与 QMT 部署目录
python -c "import shutil; shutil.copytree(r'd:\quantitative\xtquant_big_convert\src\bigqmt_signal_trader', r'd:\quantitative\chanlun-pro\src\bigqmt_signal_trader', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__')); shutil.copytree(r'd:\quantitative\xtquant_big_convert\src\bigqmt_signal_trader', r'D:\国金证券QMT交易端\python\bigqmt_signal_trader', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))"
# 3. QMT 端停止并重新运行 BIGQMT_REDIS_DRYRUN 策略后生效
```

---

## 3. 部署步骤（从零开始）

### 3.1 QMT 交易端（服务端）

1. 安装大 QMT 交易端（本机：国金证券，`D:\国金证券QMT交易端`），登录资金账号
2. 复制文件到 `D:\国金证券QMT交易端\python\`：
   - fork 仓库 `src/bigqmt_signal_trader/` → `python\bigqmt_signal_trader\`
   - fork 仓库 `src/BIGQMT_REDIS_DRYRUN.py`、`src/bigqmt_signal_trader_strategy.py`、`src/bigqmt_signal_trader_redis_rpc_runtime.py` → `python\`
   - 参照 `bigqmt_signal_trader_local_config.py` 模板，写入 `BIGQMT_ACCOUNT_ID` 与 `BIGQMT_REDIS_CONFIG`（`"transport": "zmq"`）
3. QMT → 策略编辑器 → 加载 `BIGQMT_REDIS_DRYRUN.py` → 运行
4. 就绪标志：QMT 输出面板出现 `[bigqmt_rpc] zmq started bound=tcp://127.0.0.1:15621`

### 3.2 chanlun-pro（客户端）

```powershell
cd d:\quantitative\chanlun-pro
# 桥接包（不入库）
Copy-Item -Recurse d:\quantitative\xtquant_big_convert\src\bigqmt_signal_trader src\
# 客户端配置（含资金账号，不入库）
Copy-Item src\bigqmt_signal_trader_client_config.py.example src\bigqmt_signal_trader_client_config.py
#   编辑 BIGQMT_ACCOUNT_ID / formula_server.qmt_root
# 依赖
script\bin\uv.exe sync
# 启动
.\windows_run.bat
```

`src/bigqmt_signal_trader_client_config.py` 关键项：

```python
BIGQMT_ACCOUNT_ID = "8886136661"          # 与 QMT 端 local_config 保持一致
BIGQMT_REDIS_CONFIG = {
    "transport": "zmq",                    # zmq 无需 Redis 服务器
    "formula_server": {"enabled": True, "qmt_root": r"D:\国金证券QMT交易端"},
    "rpc_timeout_seconds": 6,              # 默认 RPC 超时；长拉取由兼容层自行覆盖（120s/30s）
    "zmq": {"handler_workers": 4},         # 服务端线程池（QMT 端配置生效），0 = 关闭
}
```

### 3.3 数据同步（可选）

```powershell
python script/crontab/reboot_sync_qmt_stock_klines.py     # A股全市场入库
python script/crontab/reboot_sync_qmt_futures_klines.py   # 期货主力连续入库
# 环境变量：CL_SYNC_LIMIT=100（限只数）、CL_SYNC_WORKERS=4（并发）
```

首次全量较慢（服务端逐只下载历史），之后增量单只约 25s/4 周期。

### 3.4 验证

```powershell
# 依赖完整性
.venv\Scripts\python.exe -c "import pymysql, zmq, chanlun; print('ok')"
# tick（QMT 端需运行中）
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from bigqmt_signal_trader import xtquant_compat; print(list(xtquant_compat.xtdata.get_full_tick(['000001.SH']).keys()))"
# K 线（首次全量 1m 可达 2 分钟，属正常）
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from bigqmt_signal_trader import xtquant_compat; d=xtquant_compat.xtdata.get_market_data_ex([], ['000001.SH'], '1m', count=100, dividend_type='front', fill_data=False, timeout_seconds=120); print(len(d['000001.SH']))"
```

---

## 4. 常见问题（本次修复过的报错）

| 现象 | 原因 | 解决 |
|---|---|---|
| `No module named 'zmq'` / `pyzmq is required` | pyzmq 不在依赖，`uv run` 自动同步时卸载 | 已入 `pyproject.toml`，`uv sync` |
| `No module named 'pymysql'`（同步脚本） | 用了系统 Python 而非 `.venv` | 脚本已自动 re-exec 到 venv |
| `zmq rpc timeout: get_market_data_ex`（同步/监控） | 显式超时被 `_call` 吞掉，全量拉取按 6s 判死 | fork 已修复 + 拉取超时 120s |
| `/ticks` 500，耗时 23~145s | 客户端共享 DEALER 锁排队 + 服务端 router 串行执行慢 handler | fork 已修复（独立 socket + handler 池）；视图兜底返回空数据 |
| `/tv/history` 500 `NoneType has no len()` | QMT 无数据时 `klines=None` 直传计算 | 已判空返回 `no_data` |
| playwright `Executable doesn't exist` | chromium 未安装 | `python -m playwright install chromium`（国内镜像若无该版本走官方源） |
| `ZMQ_BIND_CONFLICT 端口被占用` | 上一策略实例未正常停止 | QMT 停止旧策略重跑，或等 60s |
| K 线只有当天数据 | 部分 QMT 版本仅注入单股下载全局（上游 #54） | 已随 v0.2.3 修复部署 |

## 5. 运维要点

- **改服务端代码后必须重启 QMT 策略**（停止 BIGQMT_REDIS_DRYRUN 再运行）；改客户端代码重启 `windows_run.bat` 即可
- 长历史拉取期间 UI tick 可能短暂返回空数据（服务端在并行处理），数据预热后消失
- 大 QMT 内置 Python 无独立行情数据服务，历史补充靠 `download_history_data` RPC 或 QMT「数据管理→补充数据」UI
- 期货账号（809222890 → 15650 端口）需在 QMT 端登录对应账号后才有该端口的 RPC 服务
- fork 源码即文档：`README.md`（架构与配置全集）、`CHANGELOG.md`（每个修复的来龙去脉）、`docs/`
