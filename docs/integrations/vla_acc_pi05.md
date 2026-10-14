# VLA-ACC Pi05 场景身份接口

VLA-ACC 的 Pi05/RoboTwin manifest runner 在应用 scene override 后，需要对
当前仿真场景重新计算 fingerprint，并与 frozen manifest 核对。RoboTwin 提供
场景读取与序列化接口；模型加载、常驻推理、cache route 和 FlashRT kernels
分别由 VLA-ACC 与 Sparse-FlashRT 提供。

## API

接口从 script.eval_policy 导出：

    from script.eval_policy import (
        _capture_frozen_scene_snapshot,
        _scene_snapshot_fingerprint,
        _frozen_snapshot_sha256,  # 历史调用名
    )

在 setup_demo、approved scene override 和随机场景初始化完成后、策略动作
开始前调用 capture。snapshot 包含 schema_version、具名 actors 的位置/
四元数，以及具名 articulations 的 root pose。无法访问 scene actor API、
场景没有可用具名实体、pose 不完整或含非有限数值时会明确抛错。

    snapshot = _capture_frozen_scene_snapshot(task_env)
    fingerprint = _scene_snapshot_fingerprint(snapshot)

_frozen_snapshot_sha256 是兼容旧 evaluator 的别名，同样对传入 snapshot
计算 SHA-256；不得将 manifest 中的 fingerprint 传回或直接当成实时结果。

## Fingerprint 兼容规则

沿用现有 frozen-manifest 规则：JSON 用 UTF-8 编码，ensure_ascii=False、
sort_keys=True、紧凑分隔符 (",", ":")，再计算 SHA-256。字典键排序，
但 actors/articulations 数组保持 SAPIEN 返回次序。列表次序属于历史输入身份
的一部分；本接口不自动重排列表，以免既有 manifest/cache 的 fingerprint
被静默改变。若将来需要顺序无关的 schema，必须版本化并显式迁移 manifest。

RoboTwin 负责提供可序列化的真实 snapshot；VLA-ACC 负责将其与对应 frozen
manifest 比较。seed 相同不替代 snapshot/fingerprint 校验。

## 测试

纯 CPU fake-scene 契约测试：

    pytest -q tests/test_frozen_scene_contract.py

该测试不启动 SAPIEN、CUDA、模型或 FlashRT，不代表真实 renderer 下场景
fingerprint 已通过。集成验证应使用 VLA-ACC 锁定的 RoboTwin commit 和真实
frozen case manifest。
