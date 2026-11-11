本轮统一实验与论文修订入口（2026-11-07）

1. 当前入口
config.json 是本轮噪声、候选预算、维度、重复次数和融合校准的配置来源。
unified_pipeline.py 是查询变换、CCADPE候选规则与融合的共用实现。
历史 duetdpe_experiments 目录保留原貌，不再把旧 rerun_revised_main.py 等入口
产生的结果当作本轮统一实验。新稿使用上一版绑定OpenReview最终PDF的修订稿
作为来源，修改日志为 manuscript_edits.json，旧稿与最终上传版均不覆盖。

2. 环境和外部数据
实际执行使用 Codex bundled Python 3.12，历史项目的 .deps 与 .gpu_deps
提供兼容的数值库和CUDA PyTorch。协议模块通过runtime.py定位依赖。
原始语料、固定编码、IVF/词法坐标索引和FTS5库来自：
<WORKSPACE>/duetdpe_experiments
代码包不包含这些大型数据、模型权重、临时owner密钥或受保护记录库。
环境版本、源代码指纹和输入范围见 quality_manifest.json、各子目录manifest。
NQ直接BM25的原生评分在最终补齐查询种子/单查询融合特征元数据之前完成，
其quality.json保留当时配置SHA 2a30d939...，没有改写成最终SHA 149fb6ff...。
该控制不使用查询噪声或融合特征；planner候选、BM25参数与切分未变。
后续三库融合控制均绑定最终配置，并逐项核验共享语义排名、query ID与holdout。

3. 质量复现流程
为保护已完成结果，请在新的独立输出目录复制本轮脚本/config，并相应调整
config.output_root；保留original_root指向只读历史数据。不要在已有结果上
删除或覆盖后反复挑选更好结果。以下为阶段入口，默认全部数据集：

run_direct_bm25.py
  建立共享全planner-1000候选精确TF缓存，运行直接300/1000两档BM25。
run_quality_stage.py --stage semantic
run_quality_stage.py --stage lexical
run_quality_stage.py --stage bm25
run_quality_stage.py --stage fusion
run_direct_fusion.py
summarize_quality_controls.py

本次Hot/MS语义阶段为消除随机读盘瓶颈，把已生成的新候选保留后采用
run_semantic_rerank_io.py顺序加载原生向量。检索候选/评分规则不变；已完成的
NQ没有重跑。中止与恢复的实际边界有记录，独立FP64精排复核通过。
53,511个query-repeat的词法顺序、cap和集合核验见quality_final_verification.json。
这些离线批处理耗时均不代替完整协议时延。

4. ppRAG独立仿真
pprag/run_controlled_pprag.py --dataset nq（其他数据集同理）
pprag/run_controlled_pprag.py --dataset nq --serial-timing
仅在GPU/内存/磁盘资源空闲时运行serial-timing。质量执行版本另存为
pprag/run_controlled_pprag_executed_quality.py并与每库manifest绑定。
算法/源码检索依据、实现差异、理想原生FP32精排假设及噪声敏感性见
pprag/SOURCE_COMPARISON.txt和pprag/summary.json。
没有把没有DP的CAPRISE消融叫完整ppRAG；也没有声称找到官方源码。

5. 完整流程计时
详见protocol/ARCHITECTURE_AND_RUN.txt。
流程为离线prepare -> 隔离服务端FTS索引 -> cached输入一致性validate ->
独占资源运行loopback。正式测量包括真实query encoder、Search与Fetch两次
TCP RPC、实际AES-GCM payload/join/document解密、原生精排及冻结融合模型。
不包含LLM生成，不代表WAN或生产安全隔离。每库16个不同query，2轮完整
warmup及3轮measurement；记录和字节实测，不以公式代替。

server_indexes仅在新目录创建。NQ使用只读backup后在已检查副本中删除docmap
并VACUUM；Hot/MS采用更保守的纯导出方式，只复制live FTS shadow/vocab，
从未复制docmap，也不执行DROP/VACUUM。四张FTS shadow表全部主键和值与原库
逐项相同，原库完整SHA和mtime未变。副本只用于测量server，owner仍持原资料。

6. 数值与统计边界
质量中的语义原生评分是FP32点积，没有重新归一化缓存向量。
CPU和GPU的FP32局部cutoff可能在很小的分差处交换候选；验证报告保留实际
差异及局部GPU重放证据，不称所有候选逐位一致。
正式请求真实重算CPU encoder；与历史CUDA编码缓存的向量/排名差异在计时
结束后单独诊断，绝不把缓存替代encoder。
质量置信区间按相同outer query配对，在每query先平均3次噪声重复，然后
做5000次bootstrap；它条件于既定校准模型，不含重新抽训练集的不确定性。

7. 阅读结果
experiment_summary.json：所有主质量、配对对照、ppRAG及完整流程计时汇总。
实验结果与修改说明.txt：中文结果解释及论文修改范围。
quality_comparison.json：同预算/更大预算直接BM25及同流程融合的配对结果。
manuscript/DuetER_unified_20261107.tex + unified_results.tex：新论文源码。
最终PDF和源码压缩包单独放在本目录。
本轮没有上传OpenReview或同步匿名代码站；摘要中已有链接标为Prior code。
protocol/PROTOCOL_TIMING_SUMMARY.json是全部9组完成后的最终计时状态。
较早验证快照中的pending描述保留其当时状态，不代表当前仍未完成。

重要结论：不能把相对semantic-only的混合检索收益归因于CCADPE词法筛选。
本轮直接BM25-300同流程融合在三库的均值均略高，HotpotQA差异CI不跨0。
这两档直接BM25是同一预算FTS5候选流程的控制，不冒称无预算的全词项全局
BM25实验。ppRAG数值组件时延也不与完整DuetER协议时延计算加速比。
