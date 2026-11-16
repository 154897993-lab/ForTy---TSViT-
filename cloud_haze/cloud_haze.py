# -*- coding: utf-8 -*-
"""
物理仿真退化引擎（云雾）——AI Earth-X 森林 第一篇论文
====================================================

任务来源：第一篇论文项目清单 谢金池任务 1（主线，唯一代码任务）
设计决策：11/12 整合版（默认加性混合；乘法受约束变体降为消融组，NDVI 严格不变）

输入格式（与 ForTy v1 TFRecord 解析结果一致）：
    s2   : float 数组, shape = (T, H, W, C)，典型 (4, 128, 128, 10)，已 /10000 归一化
    label: int64 数组, shape = (H, W)，9 类含 unknown=0（引擎不修改标签，只原样传回）

波段顺序（DeepSatModels/TSViT 惯例，红/近红外索引可在构造函数覆盖）：
    0:B2 蓝  1:B3 绿  2:B4 红  3:B5 红边  4:B6 红边  5:B7 红边
    6:B8 近红外  7:B8A 窄近红外  8:B11 短波红外1  9:B12 短波红外2

两个变体（variant 开关）：
    additive       默认。x' = (1-b)*x + b*c，b 为逐像素混合系数，c 为波段相关云亮度向量。
                   物理上对应大气散射 + 云顶反射混合，会改变 NDVI（偏差统计落表）。
    multiplicative 受约束变体。x' = a*x，同一系数乘所有波段 → NDVI 严格不变（解析性质，
                   自测逐像素验证）。物理上对应信号衰减型退化，作物理约束消融对照。

档位预设（level：'light' / 'medium' / 'heavy'；augment(level=None) 为恒等不退化）：
    light  : 云覆盖 20%，云核混合强度 0.30，雾幅度 0.02（反射率增量上限）
    medium : 云覆盖 35%，云核混合强度 0.50，雾幅度 0.04
    heavy  : 云覆盖 55%，云核混合强度 0.75，雾幅度 0.06
    云覆盖率可单独传参覆盖（合法范围 0.10-0.70）。
    雾模型：加性变体用波段相关路径辐射小增量（低强度扰动模拟轻雾），
            乘法变体用均匀信号衰减（保持 NDVI 不变性质）。

可复现性：所有随机性来自显式 seed，使用局部 numpy Generator，不污染全局随机态；
同 seed + 同参数 + 同输入 → 输出逐像素一致（自测验证）。
mask 生成只有一个入口（_build_season_masks），训练增强与评估导出走同一条路，
保证 张正锋（训练调用）与 阮业诚（评估协议）看到的退化严格一致。

跨季接口（C 组复用）：
    degrade(..., mask=m)            外部掩码直传（评估协议固定图样）
    degrade(..., overlap=True)      多季共用基准掩码，overlap_shift 控制平移扰动
    iou(m1, m2)                     掩膜 IoU 核对（C 组要求 >= 0.60）

版本：cloud_haze v1.0.0-20261113（改动请升版本号并记录，任务书原则 8）
作者：谢金池（AI 辅助实现）
"""

import numpy as np

__version__ = "1.0.0-20261113"

# ---------------------------------------------------------------- 常量与档位

# 云的波段相关亮度向量（10 波段：B2 B3 B4 B5 B6 B7 B8 B8A B11 B12）
# 可见光高反射、近红外略降、短波红外明显下降（积云典型光谱形状），量级为反射率
CLOUD_COLOR = np.array([0.80, 0.78, 0.75, 0.70, 0.66, 0.62, 0.58, 0.55, 0.38, 0.30],
                       dtype=np.float64)

# 雾的波段相关形状（加性变体用：模拟大气路径辐射，可见光为主、短波红外弱；
# 数值 = 相对幅度，实际增量 = haze_strength × 场值 × 本形状）
HAZE_SHAPE = np.array([1.00, 1.00, 0.95, 0.85, 0.80, 0.75, 0.60, 0.58, 0.35, 0.25],
                      dtype=np.float64)

LEVEL_PRESETS = {
    # 云覆盖率 / 云核混合强度上限(加性)或衰减深度(乘法) / 雾幅度（加性=反射率增量上限，乘法=衰减幅度）
    "light":  dict(cloud_coverage=0.20, cloud_strength=0.30, haze_strength=0.02),
    "medium": dict(cloud_coverage=0.35, cloud_strength=0.50, haze_strength=0.04),
    "heavy":  dict(cloud_coverage=0.55, cloud_strength=0.75, haze_strength=0.06),
}

MIN_COVERAGE, MAX_COVERAGE = 0.10, 0.70
MULT_FLOOR = 0.25        # 乘法变体云核最低透射率 a_min：x' = a*x, a ∈ [0.25, 1]
C_OVERLAP_IOU_MIN = 0.60  # C 组跨季掩膜 IoU 合同下限（任务书：掩膜 IoU ≥ 0.60）


# ---------------------------------------------------------------- 工具函数

def iou(mask_a, mask_b):
    """两个二值掩码的 IoU（C 组跨季重叠核对用，任务书要求 >= 0.60）"""
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / float(union)


def ndvi(s2, red_idx=2, nir_idx=6, eps=1e-8):
    """逐像素 NDVI。s2: (..., C) 任意前面维度，返回去掉通道维的数组"""
    red = s2[..., red_idx].astype(np.float64)
    nir = s2[..., nir_idx].astype(np.float64)
    return (nir - red) / (nir + red + eps)


def ndvi_bias_stats(before, after, mask=None, red_idx=2, nir_idx=6):
    """
    云雾区 NDVI 偏差统计（任务书：中位数与 P95 落表）。
    before/after: (T,H,W,C)；mask: (H,W) 二值，None 表示全图。
    返回 dict(median, p95)，基于 |ΔNDVI|。
    """
    d = np.abs(ndvi(after, red_idx, nir_idx) - ndvi(before, red_idx, nir_idx))
    if mask is not None:
        m = np.broadcast_to(np.asarray(mask).astype(bool), d.shape)
        d = d[m]
    else:
        d = d.ravel()
    if d.size == 0:
        return dict(median=0.0, p95=0.0)
    return dict(median=float(np.median(d)), p95=float(np.percentile(d, 95)))


def _blur_1d(v, r):
    pad = np.pad(v, r, mode="edge")
    kernel = np.ones(2 * r + 1)
    return np.convolve(pad, kernel, mode="valid") / (2 * r + 1)


def _box_blur(img, r):
    """可分离盒式模糊（边界复制），连跑多次近似高斯"""
    out = img.copy()
    for axis in (0, 1):
        out = np.apply_along_axis(_blur_1d, axis, out, r)
    return out


def _smooth_noise_field(rng, size=128, n_octaves=3, base=8):
    """
    生成 [0,1] 平滑随机场（多尺度上采样噪声叠加 + 盒式模糊），云掩码软边界来源。
    小分辨率白噪声 np.kron 放大 → 多尺度叠加 → 三次模糊 → 归一化。
    """
    field = np.zeros((size, size), dtype=np.float64)
    for k in range(n_octaves):
        res = base * (2 ** k)
        small = rng.random((res, res))
        block = max(1, size // res)
        up = np.kron(small, np.ones((block, block)))[:size, :size]
        field += up
    for _ in range(3):
        field = _box_blur(field, 2)
    field -= field.min()
    if field.max() > 0:
        field /= field.max()
    return field


def _binary_mask_from_field(field, coverage):
    """
    由软场按分位数取阈值 → (soft, binary)，binary 覆盖率精确逼近 coverage。
    soft 只在掩码内部保留软值，边界渐变供混合系数使用。
    """
    tau = np.quantile(field, 1.0 - coverage)
    binary = field > tau
    if not binary.any():  # 极端保护
        binary = field >= field.max()
    soft = np.where(binary, field, 0.0)
    return soft, binary


def _season_mask_seed(seed, s):
    """每季掩码的确定性种子推导（训练/评估两条路径共用，保证一致）"""
    return int(seed) * 1009 + int(s) * 7 + 1


def _rng_shift(rng, max_shift):
    """跨季掩码平移量：各轴随机 ±max_shift（0 时不动 → IoU=1）"""
    if max_shift <= 0:
        return (0, 0)
    return (int(rng.integers(-max_shift, max_shift + 1)),
            int(rng.integers(-max_shift, max_shift + 1)))


# ---------------------------------------------------------------- 核心引擎

class CloudHazeEngine:
    """
    用法（张正锋·训练增强侧）：
        engine = CloudHazeEngine()
        s2_deg, label, meta = engine.augment(s2, label, seed=0, level="medium",
                                             variant="additive")

    用法（阮业诚·评估协议侧，固定掩码）：
        masks = engine.export_masks(s2.shape, seed=0, level="medium")   # {季节: 掩码dict}
        s2_deg, meta = engine.degrade(s2, seed=0, level="medium",
                                      mask=masks[2]["soft"])
        跨季重叠：degrade(s2, seed=0, level="medium", seasons=[0, 2], overlap=True,
                          overlap_shift=8) → 两季掩码 IoU 用 iou() 核对 >= 0.60
    """

    def __init__(self, red_idx=2, nir_idx=6, cloud_color=None):
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        self.cloud_color = (CLOUD_COLOR if cloud_color is None
                            else np.asarray(cloud_color, dtype=np.float64))

    # ---------------- 参数解析 ----------------

    def _resolve(self, level, cloud_coverage, cloud_strength, haze_strength):
        """档位预设 + 显式参数覆盖 → 完整参数 dict"""
        if level not in LEVEL_PRESETS:
            raise ValueError(f"level 须为 light/medium/heavy，收到 {level!r}")
        p = dict(LEVEL_PRESETS[level])
        if cloud_coverage is not None:
            if not (MIN_COVERAGE <= cloud_coverage <= MAX_COVERAGE):
                raise ValueError(f"cloud_coverage 须在 [{MIN_COVERAGE},{MAX_COVERAGE}]，"
                                f"收到 {cloud_coverage}")
            p["cloud_coverage"] = float(cloud_coverage)
        if cloud_strength is not None:
            p["cloud_strength"] = float(cloud_strength)
        if haze_strength is not None:
            p["haze_strength"] = float(haze_strength)
        return p

    @staticmethod
    def _pick_seasons(seasons, t_len, rng):
        if seasons is None:
            return list(range(t_len))                    # 默认全部季节受扰
        if isinstance(seasons, (int, np.integer)):
            k = int(min(max(1, seasons), t_len))
            return sorted(int(x) for x in rng.choice(t_len, size=k, replace=False))
        return [int(s) for s in seasons]                 # 显式指定季节列表

    # ---------------- 掩码（唯一生成入口） ----------------

    def _build_season_masks(self, seed, shape_thwc, p, seasons, mask, overlap,
                            overlap_shift):
        """
        构建各受扰季节的掩码。优先级：外部 mask > overlap 共用基准 > 每季独立。
        返回 dict{season: dict(soft, binary, coverage_actual, shift)}。
        训练 degrade 与评估 export_masks 都走本方法 → 掩码严格同源。
        """
        h, w = shape_thwc[1], shape_thwc[2]
        result = {}

        if mask is not None:                             # 外部掩码：全部受扰季共用
            ext = np.asarray(mask, dtype=np.float64)
            if ext.shape != (h, w):
                raise ValueError(f"mask shape {ext.shape} 与图像 ({h},{w}) 不符")
            if ext.max() > 1.0:
                ext = ext / max(ext.max(), 1e-8)
            binary = ext > 0.5
            for s in seasons:
                result[s] = dict(soft=ext, binary=binary,
                                 coverage_actual=float(binary.mean()), shift=(0, 0))
            return result

        base = self._make_mask(seed, (h, w), p["cloud_coverage"])
        if not overlap:                                  # 每季独立掩码（B 组独立采样）
            for s in seasons:
                result[s] = dict(self._make_mask(_season_mask_seed(seed, s),
                                                  (h, w), p["cloud_coverage"]))
                result[s]["shift"] = (0, 0)
            return result

        for i, s in enumerate(seasons):                  # C 组：共用基准 + 平移扰动
            if i == 0:
                result[s] = dict(base, shift=(0, 0))
                continue
            sh = _rng_shift(np.random.default_rng(int(seed) + s), overlap_shift)
            b0 = base["binary"]
            while sh != (0, 0):
                rolled = np.roll(b0, sh, axis=(0, 1))
                if iou(b0, rolled) >= C_OVERLAP_IOU_MIN:
                    break
                sh = (int(sh[0] / 2), int(sh[1] / 2))  # IoU 不足则平移减半，直至满足合同
            rolled_soft = np.roll(base["soft"], sh, axis=(0, 1))
            rolled_b = np.roll(b0, sh, axis=(0, 1))
            result[s] = dict(
                soft=rolled_soft, binary=rolled_b,
                coverage_actual=float(rolled_b.mean()), shift=sh)
        return result

    @staticmethod
    def _make_mask(seed, shape_hw, coverage, coverage_jitter=0.03):
        """
        生成一次云掩码。返回 dict(soft, binary, coverage_target, coverage_actual)。
        coverage_jitter：实际覆盖率在目标 ±jitter 内随机浮动（避免全部样本覆盖率完全一致）。
        """
        rng = np.random.default_rng(seed)
        h, w = shape_hw
        target = float(np.clip(coverage + rng.uniform(-coverage_jitter, coverage_jitter),
                               MIN_COVERAGE, MAX_COVERAGE))
        field = _smooth_noise_field(rng, size=min(h, w))
        if field.shape != (h, w):
            field = np.broadcast_to(field, (h, w))
        soft, binary = _binary_mask_from_field(field, target)
        return dict(soft=soft, binary=binary,
                    coverage_target=target, coverage_actual=float(binary.mean()))

    def export_masks(self, s2_shape, seed, level="medium", seasons=None,
                     cloud_coverage=None, overlap=False, overlap_shift=0):
        """
        导出全部受扰季节的掩码（评估协议固定图样，不执行退化）。
        与 degrade(..., 同参数) 内部使用的掩码严格一致。
        """
        p = self._resolve(level, cloud_coverage, None, None)
        t_len = int(s2_shape[0])
        seasons_idx = self._pick_seasons(seasons, t_len, np.random.default_rng(seed))
        return self._build_season_masks(seed, s2_shape, p, seasons_idx,
                                        None, overlap, overlap_shift)

    # ---------------- 核心退化 ----------------

    def degrade(self, s2, seed, level="medium", variant="additive",
                cloud_coverage=None, cloud_strength=None, haze_strength=None,
                seasons=None, mask=None, overlap=False, overlap_shift=0,
                return_meta=True):
        """
        对光学时序施加云雾退化。只改 s2 数值，不触碰标签（标签由 augment() 原样传回）。

        参数
        ----
        s2 : (T,H,W,C) float 数组，ForTy /10000 反射率口径
        seed : int，随机种子（可复现关键）
        level : 'light'/'medium'/'heavy'
        variant : 'additive'（默认，含云颜色）| 'multiplicative'（NDVI 严格不变，消融变体）
        cloud_coverage : None=档位默认；显式传入覆盖（0.10-0.70）
        cloud_strength / haze_strength : None=档位默认
        seasons : None=全部季节独立采样 | int k=随机 k 季 | list=指定季节索引
        mask : (H,W) 外部掩码（评估协议固定图样；所有受扰季共用）
        overlap : True 时无外部 mask 也让多季共用基准掩码（C 组）
        overlap_shift : 跨季平移像素上限；若平移后 IoU < 0.60 引擎自动将平移量
                        逐次减半直至满足合同（最终平移量记录在 meta.per_season[].shift）

        返回
        ----
        s2_deg : (T,H,W,C) float32，与输入同 shape
        meta : dict（return_meta=False 时只返回数组）
        """
        if variant not in ("additive", "multiplicative"):
            raise ValueError(f"variant 须为 additive/multiplicative，收到 {variant!r}")
        s2 = np.asarray(s2)
        if s2.ndim != 4:
            raise ValueError(f"s2 须为 (T,H,W,C) 4 维，收到 shape={s2.shape}")
        p = self._resolve(level, cloud_coverage, cloud_strength, haze_strength)

        out = s2.astype(np.float32).copy()
        t_len, h, w, _ = out.shape
        seasons_idx = self._pick_seasons(seasons, t_len, np.random.default_rng(seed))

        masks = self._build_season_masks(seed, out.shape, p, seasons_idx,
                                         mask, overlap, overlap_shift)

        # 全图雾（仅受扰季施加；haze_strength=0 可关；雾种子 = seed*31+17+season，可复现）
        season_metas = {}
        cloud_pix_total = 0
        for s in seasons_idx:
            m = masks[s]
            soft = np.asarray(m["soft"], dtype=np.float64)
            binary = np.asarray(m["binary"], dtype=bool)
            smax = max(soft.max(), 1e-8)
            beta = np.clip(soft / smax, 0.0, 1.0) * p["cloud_strength"]

            frame = out[s].astype(np.float64)             # (H,W,C)
            if variant == "additive":
                mixed = ((1.0 - beta[..., None]) * frame
                         + beta[..., None] * self.cloud_color[None, None, :])
                if p["haze_strength"] > 0:               # 加性雾：路径辐射小增量
                    hf = _smooth_noise_field(
                        np.random.default_rng(int(seed) * 31 + 17 + s), size=min(h, w))
                    if hf.shape != (h, w):
                        hf = np.broadcast_to(hf, (h, w))
                    hb = p["haze_strength"] * (0.6 + 0.4 * hf)
                    mixed = mixed + hb[..., None] * HAZE_SHAPE[None, None, :]
            else:  # multiplicative：全程逐像素全波段同乘 → NDVI 严格不变
                a = 1.0 - (beta / max(p["cloud_strength"], 1e-8)) * (1.0 - MULT_FLOOR)
                if p["haze_strength"] > 0:               # 乘法雾：均匀信号衰减
                    hf = _smooth_noise_field(
                        np.random.default_rng(int(seed) * 31 + 17 + s), size=min(h, w))
                    if hf.shape != (h, w):
                        hf = np.broadcast_to(hf, (h, w))
                    a = a * (1.0 - p["haze_strength"] * (0.6 + 0.4 * hf))
                mixed = a[..., None] * frame

            out[s] = np.clip(mixed, 0.0, None).astype(np.float32)
            cloud_pix_total += int(binary.sum())
            season_metas[s] = dict(coverage_actual=float(binary.mean()),
                                   n_cloud_pixels=int(binary.sum()),
                                   shift=tuple(m.get("shift", (0, 0))))

        meta = dict(
            engine=f"cloud_haze v{__version__}",
            variant=variant, level=level, seed=int(seed),
            cloud_coverage_target=p["cloud_coverage"],
            cloud_strength=p["cloud_strength"], haze_strength=p["haze_strength"],
            seasons_disturbed=seasons_idx,
            n_cloud_pixels_total=cloud_pix_total,
            per_season=season_metas,
            red_idx=self.red_idx, nir_idx=self.nir_idx,
            binary_masks={s: masks[s]["binary"] for s in seasons_idx},
        )
        return (out, meta) if return_meta else out

    # ---------------- 便捷入口（训练侧） ----------------

    def augment(self, s2, label=None, seed=0, level="medium", variant="additive", **kw):
        """
        训练增强入口：退化 s2，标签原样传回（引擎不修改标签——验收标准 2）。
        level=None → 原图直返（验收标准 1：强度 0 恒等）。
        返回 (s2_out, label, meta)。
        """
        if level in (None, "none", 0):
            s2_out = np.asarray(s2).copy()
            meta = dict(identity=True, engine=f"cloud_haze v{__version__}",
                        seed=int(seed))
            return s2_out, label, meta
        s2_out, meta = self.degrade(np.asarray(s2), seed=seed, level=level,
                                    variant=variant, **kw)
        return s2_out, label, meta
