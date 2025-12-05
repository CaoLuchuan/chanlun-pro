# -*- coding: utf-8 -*-
import datetime
import copy
from typing import List, Union, Dict, Tuple

import numpy as np
import pandas as pd
import talib

from chanlun import cl_interface
from chanlun import cl_utils
from chanlun.cl_interface import (
    ICL, Config, Kline, CLKline, FX, BI, XD, ZS, MMD, BC, LINE, TZXL, XLFX,
    query_macd_ld, compare_ld_beichi, user_custom_mmd
)
from chanlun.cl_utils import cal_macd_bis_is_bc


class CL(ICL):
    def __init__(
        self,
        code: str,
        frequency: str,
        config: Union[dict, None] = None,
        start_datetime: datetime.datetime = None,
    ):
        self.code = code
        self.frequency = frequency
        self.config = cl_utils.query_cl_chart_config("common", code)
        if config:
            self.config.update(config)
        self.start_datetime = start_datetime

        self.klines: List[Kline] = []
        self.cl_klines: List[CLKline] = []
        self.idx: Dict[str, Dict] = {"macd": {"dif": [], "dea": [], "hist": []}}

        self.fxs: List[FX] = []
        self.bis: List[BI] = []
        self.xds: List[XD] = []
        self.zsds: List[XD] = []
        self.qsds: List[XD] = []

        self.bi_zss: Dict[str, List[ZS]] = {}
        self.xd_zss: Dict[str, List[ZS]] = {}
        self.zsd_zss: List[ZS] = []
        self.qsd_zss: List[ZS] = []

        self.last_bi_zs: Union[ZS, None] = None
        self.last_xd_zs: Union[ZS, None] = None
        
        # 笔确认机制相关状态
        self.pending_bis: List[BI] = []  # 待确认的笔列表

    def get_code(self) -> str:
        return self.code

    def get_frequency(self) -> str:
        return self.frequency

    def get_config(self) -> dict:
        return self.config

    def get_src_klines(self) -> List[Kline]:
        return self.klines

    def get_klines(self) -> List[Kline]:
        if self.config["kline_type"] == Config.KLINE_TYPE_CHANLUN.value:
            return self.cl_klines
        return self.klines

    def get_cl_klines(self) -> List[CLKline]:
        return self.cl_klines

    def get_idx(self) -> dict:
        return self.idx

    def get_fxs(self) -> List[FX]:
        return self.fxs

    def get_bis(self) -> List[BI]:
        return self.bis

    def get_xds(self) -> List[XD]:
        return self.xds

    def get_zsds(self) -> List[XD]:
        return self.zsds

    def get_qsds(self) -> List[XD]:
        return self.qsds

    def get_bi_zss(self, zs_type: str = None) -> List[ZS]:
        if zs_type is None:
            zs_type = self.config["zs_bi_type"][0]
        return self.bi_zss.get(zs_type, [])

    def get_xd_zss(self, zs_type: str = None) -> List[ZS]:
        if zs_type is None:
            zs_type = self.config["zs_xd_type"][0]
        return self.xd_zss.get(zs_type, [])

    def get_zsd_zss(self) -> List[ZS]:
        return self.zsd_zss

    def get_qsd_zss(self) -> List[ZS]:
        return self.qsd_zss

    def get_last_bi_zs(self) -> Union[ZS, None]:
        return self.last_bi_zs

    def get_last_xd_zs(self) -> Union[ZS, None]:
        return self.last_xd_zs

    def process_klines(self, klines: pd.DataFrame):
        if len(klines) == 0:
            return

        new_klines = []
        if "date" not in klines.columns:
             if isinstance(klines.index, pd.DatetimeIndex):
                 klines = klines.reset_index()
                 klines.rename(columns={"index": "date"}, inplace=True)

        for _, row in klines.iterrows():
            date_val = row["date"]
            if isinstance(date_val, str):
                date_val = pd.to_datetime(date_val)
            
            if self.start_datetime and date_val < self.start_datetime:
                continue

            # Handle different volume column names
            volume = 0
            if "volume" in row:
                volume = float(row["volume"])
            elif "vol" in row:
                volume = float(row["vol"])
                
            k = Kline(
                index=len(self.klines) + len(new_klines),
                date=date_val,
                h=float(row["high"]),
                l=float(row["low"]),
                o=float(row["open"]),
                c=float(row["close"]),
                a=volume,
            )
            new_klines.append(k)

        if len(new_klines) == 0:
            return

        if len(self.klines) > 0:
            last_k = self.klines[-1]
            nk_dt = new_klines[0].date
            lk_dt = last_k.date
            if isinstance(nk_dt, pd.Timestamp) and isinstance(lk_dt, pd.Timestamp):
                if (nk_dt.tz is not None and lk_dt.tz is None):
                    nk_cmp = nk_dt.tz_localize(None)
                    lk_cmp = lk_dt
                elif (nk_dt.tz is None and lk_dt.tz is not None):
                    nk_cmp = nk_dt
                    lk_cmp = lk_dt.tz_localize(None)
                else:
                    nk_cmp = nk_dt
                    lk_cmp = lk_dt
            else:
                nk_cmp = nk_dt
                lk_cmp = lk_dt
            if nk_cmp == lk_cmp:
                self.klines[-1] = new_klines[0]
                new_klines = new_klines[1:]
            elif nk_cmp < lk_cmp:
                idx = -1
                for i, k in enumerate(self.klines):
                    kd = k.date
                    if isinstance(kd, pd.Timestamp) and isinstance(nk_dt, pd.Timestamp):
                        if (kd.tz is not None and nk_dt.tz is None):
                            kd_cmp = kd.tz_localize(None)
                            nkd_cmp = nk_dt
                        elif (kd.tz is None and nk_dt.tz is not None):
                            kd_cmp = kd
                            nkd_cmp = nk_dt.tz_localize(None)
                        else:
                            kd_cmp = kd
                            nkd_cmp = nk_dt
                    else:
                        kd_cmp = kd
                        nkd_cmp = nk_dt
                    if kd_cmp >= nkd_cmp:
                        idx = i
                        break
                if idx != -1:
                    self.klines = self.klines[:idx]
                    for i, k in enumerate(self.klines):
                        k.index = i
                    for i, k in enumerate(new_klines):
                        k.index = len(self.klines) + i
        
        self.klines.extend(new_klines)

        self._cal_idx()
        self._cal_cl_klines()
        self._cal_fx()
        self._cal_bi()
        self._cal_xd()
        self._cal_zsd()
        self._cal_zs()
        self._cal_mmd_bc()

    def _cal_idx(self):
        close_prices = np.array([k.c for k in self.klines])
        if len(close_prices) < 35:
             self.idx["macd"]["dif"] = [0] * len(close_prices)
             self.idx["macd"]["dea"] = [0] * len(close_prices)
             self.idx["macd"]["hist"] = [0] * len(close_prices)
             return

        dif, dea, hist = talib.MACD(
            close_prices,
            fastperiod=int(self.config["idx_macd_fast"]),
            slowperiod=int(self.config["idx_macd_slow"]),
            signalperiod=int(self.config["idx_macd_signal"]),
        )
        np.nan_to_num(dif, copy=False)
        np.nan_to_num(dea, copy=False)
        np.nan_to_num(hist, copy=False)

        self.idx["macd"]["dif"] = dif
        self.idx["macd"]["dea"] = dea
        self.idx["macd"]["hist"] = hist * 2

    def _cal_cl_klines(self):
        self.cl_klines = []
        if len(self.klines) == 0:
            return

        k0 = self.klines[0]
        ck0 = CLKline(
            k_index=k0.index,
            date=k0.date,
            h=k0.h,
            l=k0.l,
            o=k0.o,
            c=k0.c,
            a=k0.a,
            klines=[k0],
            index=0,
            _n=1
        )
        self.cl_klines.append(ck0)

        direction = None  # 初始方向未知，需要根据实际K线关系确定

        for i in range(1, len(self.klines)):
            k = self.klines[i]
            last_ck = self.cl_klines[-1]

            is_included = False
            if (k.h >= last_ck.h - 1e-7 and k.l <= last_ck.l + 1e-7) or (last_ck.h >= k.h - 1e-7 and last_ck.l <= k.l + 1e-7):
                is_included = True
            
            if is_included:
                # 缠论时间优先级原则：包含处理方向由前一个非包含K线决定
                # 如果还没有确定方向，根据价格关系确定
                if direction is None:
                    # 根据前一根K线的收盘价关系确定方向
                    direction = "up" if k.c > last_ck.c else "down"
                
                new_h = 0
                new_l = 0
                if direction == "up":
                    new_h = max(last_ck.h, k.h)
                    new_l = max(last_ck.l, k.l)  # 向上处理：取高高
                else:
                    new_h = min(last_ck.h, k.h)  # 向下处理：取低低
                    new_l = min(last_ck.l, k.l)
                
                last_ck.h = new_h
                last_ck.l = new_l
                last_ck.k_index = k.index
                last_ck.date = k.date
                last_ck.a += k.a
                last_ck.klines.append(k)
                last_ck.n += 1
            else:
                # 修复：方向判断基于时间优先级
                if k.h > last_ck.h and k.l > last_ck.l:
                    direction = "up"
                elif k.h < last_ck.h and k.l < last_ck.l:
                    direction = "down"
                else:
                    # 非标准情况，根据价格关系确定方向
                    direction = "up" if k.c > last_ck.c else "down"
                
                new_ck = CLKline(
                    k_index=k.index,
                    date=k.date,
                    h=k.h,
                    l=k.l,
                    o=k.o,
                    c=k.c,
                    a=k.a,
                    klines=[k],
                    index=len(self.cl_klines),
                    _n=1
                )
                new_ck.up_qs = direction
                self.cl_klines.append(new_ck)

    def _cal_fx(self):
        self.fxs = []
        if len(self.cl_klines) < 3:
            return

        fx_qj = self.config["fx_qj"]
        fx_bh = self.config["fx_bh"]
        
        # 首先对K线进行分型级别的包含处理
        processed_klines = self._process_fx_klines(self.cl_klines, fx_bh)
        
        for i in range(1, len(processed_klines) - 1):
            k1 = processed_klines[i-1]
            k2 = processed_klines[i]
            k3 = processed_klines[i+1]
            
            # 检查是否为有效的分型
            fx_type = None
            if k2.h > k1.h and k2.h > k3.h:
                fx_type = "ding"
            elif k2.l < k1.l and k2.l < k3.l:
                fx_type = "di"
            
            if fx_type:
                # 检查分型有效性：确保分型K线不是被包含的K线
                if hasattr(k2, '_is_included') and k2._is_included:
                    continue  # 跳过被包含的K线作为分型中心
                
                # 检查分型之间是否有足够的K线（至少1根K线间隔）
                if len(self.fxs) > 0:
                    last_fx = self.fxs[-1]
                    k_diff = abs(k2.index - last_fx.k.index)
                    if k_diff < 2:  # 至少需要1根K线间隔
                        # 保留更极值的分型
                        if fx_type == "ding" and k2.h > last_fx.val:
                            self.fxs.pop()
                        elif fx_type == "di" and k2.l < last_fx.val:
                            self.fxs.pop()
                        else:
                            continue
                
                val = 0
                if fx_type == "ding":
                    if fx_qj == Config.FX_QJ_CK.value:
                        val = k2.h
                    else:
                        val = max([_k.h for _k in k2.klines])
                else:
                    if fx_qj == Config.FX_QJ_CK.value:
                        val = k2.l
                    else:
                        val = min([_k.l for _k in k2.klines])
                
                fx = FX(
                    _type=fx_type,
                    k=k2,
                    klines=[k1, k2, k3],
                    val=val,
                    index=len(self.fxs),
                    done=True
                )
                self.fxs.append(fx)

    def check_bi_valid(self, start_fx: FX, end_fx: FX, bi_type: str) -> bool:
        # 顶底必须交替
        if start_fx.type == end_fx.type:
            return False

        # 方向必须满足：顶->低 新低；底->顶 新高
        # 严格检查：下笔的顶分型高点必须高于底分型高点（不仅仅是val，防止包含关系导致的误判）
        # 严格检查：上笔的底分型低点必须低于顶分型低点
        # 顶分型定义：中间K线高点最高。底分型定义：中间K线低点最低。
        # 这里使用 fx.val (已经根据配置取了 High/Low)
        if start_fx.type == "ding":
            if end_fx.val >= start_fx.val: return False
            if end_fx.high(Config.FX_QJ_CK.value, Config.FX_QY_THREE.value) >= start_fx.high(Config.FX_QJ_CK.value, Config.FX_QY_THREE.value): return False
        
        if start_fx.type == "di":
            if end_fx.val <= start_fx.val: return False
            if end_fx.low(Config.FX_QJ_CK.value, Config.FX_QY_THREE.value) <= start_fx.low(Config.FX_QJ_CK.value, Config.FX_QY_THREE.value): return False

        # 缠论K线中心索引差值（用于判断是否存在独立缠论K线）
        ck_diff = end_fx.k.index - start_fx.k.index

        # 原始K线数量（用于新笔/简单笔判断）
        src_k_num = 0
        for ck in self.cl_klines[start_fx.k.index : end_fx.k.index + 1]:
            src_k_num += len(ck.klines)

        if bi_type == Config.BI_TYPE_OLD.value:
            # 老笔：5根合并后的K线 (ck_diff >= 4 means 5 lines: 0, 1, 2, 3, 4)
            if ck_diff < 4:
                return False
            # 确保中间有独立K线：顶底分型各3根，共用0根时至少5根，共用1根时5根...
            # 标准老笔定义：顶分型+独立K线+底分型。
            # 顶分型占3根，底分型占3根。如果完全不共用，需要3+N+3。
            # 但缠论允许共用，只要顶底不重叠。
            # 最严格定义：顶分型区间与底分型区间不重叠。
            # ck_diff >= 4 意味着：Index 0 (Start), 1, 2, 3, 4 (End)。
            # 中间有 1, 2, 3 三根。
            # 顶分型用到 0, 1 (如果用FX_QY_MIDDLE则只看0)。
            # 通常老笔要求：顶底分型之间至少有一根不属于顶也不属于底的K线？
            # 不，老笔定义是：顶分型和底分型之间至少有一根独立的K线，或者顶分型底分型完全不共用K线。
            # ck_diff = 4: Start(0), 1, 2, 3, End(4).
            # Ding(0): [ -1, 0, 1] (假设) -> End(4): [3, 4, 5].
            # 此时 2 是独立的。所以 ck_diff >= 4 是满足“中间有独立K线”的最小条件。
            return True
        elif bi_type == Config.BI_TYPE_NEW.value:
            # 新笔：4根合并后的K线 (ck_diff >= 3 means 4 lines) 且 5根原始K线
            # 用户需求：新笔4根合并K线，5根原始K线
            if src_k_num < 5:
                return False
            return ck_diff >= 3
        elif bi_type == Config.BI_TYPE_JDB.value:
            # 简单笔：至少5根原始K线即可
            return src_k_num >= 5
        elif bi_type == Config.BI_TYPE_DD.value:
            # 顶底成笔：需满足有一根独立K线
            return ck_diff >= 4
        else:
            return ck_diff >= 4

    def _bi_high_low(self, start_fx: FX, end_fx: FX) -> Tuple[float, float]:
        qy = self.config.get("fx_qy", Config.FX_QY_THREE.value)
        qj_ck = Config.FX_QJ_CK.value
        qj_k = Config.FX_QJ_K.value
        bi_qj = self.config.get("bi_qj", Config.BI_QJ_DD.value)
        if bi_qj == Config.BI_QJ_DD.value:
            high = max(start_fx.val, end_fx.val)
            low = min(start_fx.val, end_fx.val)
        elif bi_qj == Config.BI_QJ_CK.value:
            high = max(start_fx.high(qj_ck, qy), end_fx.high(qj_ck, qy))
            low = min(start_fx.low(qj_ck, qy), end_fx.low(qj_ck, qy))
        elif bi_qj == Config.BI_QJ_K.value:
            high = max(start_fx.high(qj_k, qy), end_fx.high(qj_k, qy))
            low = min(start_fx.low(qj_k, qy), end_fx.low(qj_k, qy))
        else:
            high = max(start_fx.val, end_fx.val)
            low = min(start_fx.val, end_fx.val)
        return high, low

    def _process_fx_klines(self, klines, fx_bh):
        """分型级别的K线包含处理方法"""
        if fx_bh == "fx_bh_no":
            return klines  # 不进行包含处理
        
        processed = []
        for k in klines:
            if not processed:
                processed.append(k)
                continue
                
            last = processed[-1]
            # 包含关系判断
            is_included = (k.h >= last.h - 1e-7 and k.l <= last.l + 1e-7) or (last.h >= k.h - 1e-7 and last.l <= k.l + 1e-7)
            
            if is_included:
                # 标记被包含的K线
                k._is_included = True
                
                # 包含处理：根据趋势方向
                if fx_bh == "fx_bh_dingdi":
                    # 顶底分型分别处理 - 简化版本
                    last.h = max(last.h, k.h)
                    last.l = min(last.l, k.l)
                else:
                    # 标准包含处理：根据收盘价决定方向
                    if k.c > last.c:
                        # 向上处理：取高高
                        last.h = max(last.h, k.h)
                        last.l = max(last.l, k.l)
                    else:
                        # 向下处理：取低低
                        last.h = min(last.h, k.h)
                        last.l = min(last.l, k.l)
                last.klines.extend(k.klines)
                last.n += getattr(k, 'n', 1)
            else:
                processed.append(k)
        
        return processed
    
    def _verify_feature_sequence(self, bi: BI) -> bool:
        """
        验证笔的特征序列分型
        根据缠论原著，笔的结束需要通过特征序列的分型来确认
        """
        if bi.type == "up":
            # 向上笔的特征序列是向下笔
            return self._check_up_bi_feature_sequence(bi)
        else:
            # 向下笔的特征序列是向上笔  
            return self._check_down_bi_feature_sequence(bi)
    
    def _check_up_bi_feature_sequence(self, bi: BI) -> bool:
        """
        检查向上笔的特征序列分型
        向上笔的特征序列是向下笔，需要找到有效的底分型来确认笔结束
        """
        end_index = bi.end.k.index
        
        # 特征序列分型应该在笔结束后形成
        if end_index + 3 >= len(self.cl_klines):
            return False
            
        # 从笔结束位置开始向后查找特征序列的底分型
        # 特征序列是笔结束后的反向笔序列（向下笔）
        for i in range(end_index + 1, len(self.cl_klines) - 2):
            k1 = self.cl_klines[i]
            k2 = self.cl_klines[i+1]
            k3 = self.cl_klines[i+2]
            
            # 检查是否为有效的底分型（中间K线低点最低）
            if k2.l < k1.l and k2.l < k3.l:
                # 底分型应该确认笔的结束，所以其低点不能创新低（高于笔的起点）
                if k2.l > bi.start.val:
                    # 检查分型有效性：不能有包含关系
                    if not hasattr(k2, '_is_included') or not k2._is_included:
                        return True
                        
        return False
    
    def _check_down_bi_feature_sequence(self, bi: BI) -> bool:
        """
        检查向下笔的特征序列分型
        向下笔的特征序列是向上笔，需要找到有效的顶分型来确认笔结束
        """
        end_index = bi.end.k.index
        
        # 特征序列分型应该在笔结束后形成
        if end_index + 3 >= len(self.cl_klines):
            return False
            
        # 从笔结束位置开始向后查找特征序列的顶分型
        # 特征序列是笔结束后的反向笔序列（向上笔）
        for i in range(end_index + 1, len(self.cl_klines) - 2):
            k1 = self.cl_klines[i]
            k2 = self.cl_klines[i+1]
            k3 = self.cl_klines[i+2]
            
            # 检查是否为有效的顶分型（中间K线高点最高）
            if k2.h > k1.h and k2.h > k3.h:
                # 顶分型应该确认笔的结束，所以其高点不能创新高（低于笔的起点）
                if k2.h < bi.start.val:
                    # 检查分型有效性：不能有包含关系
                    if not hasattr(k2, '_is_included') or not k2._is_included:
                        return True
                        
        return False
    
    def _confirm_bi_with_next_bi(self, bi: BI, next_bi: BI) -> bool:
        """
        通过反向下一笔来确认前一笔的结束
        """
        # 检查笔的方向是否相反
        if bi.type == next_bi.type:
            return False
            
        # 检查下一笔是否有效确认前一笔
        if bi.type == "up":
            # 向上笔需要被向下笔确认
            # 向下笔的起点（顶分型）应该低于或等于向上笔的终点（顶分型）
            return next_bi.start.val <= bi.end.val
        else:
            # 向下笔需要被向上笔确认  
            # 向上笔的起点（底分型）应该高于或等于向下笔的终点（底分型）
            return next_bi.start.val >= bi.end.val

    def _cal_bi(self):
        self.bis = []
        self.pending_bis = []  # 清空待确认笔列表
        if len(self.fxs) < 2:
            return

        bi_type = self.config["bi_type"]
        
        # 1. Find the first valid pen
        start_fx = self.fxs[0]
        idx = 1
        
        while idx < len(self.fxs):
            end_fx = self.fxs[idx]
            
            if self.check_bi_valid(start_fx, end_fx, bi_type):
                bi = BI(
                    start=start_fx,
                    end=end_fx,
                    _type="down" if start_fx.type == "ding" else "up",
                    index=len(self.bis)
                )
                bi.high, bi.low = self._bi_high_low(start_fx, end_fx)
                
                # 初始笔不进行确认，直接加入待确认列表
                bi.feature_sequence_verified = self._verify_feature_sequence(bi)
                self.pending_bis.append(bi)
                self.bis.append(bi)
                break
            
            # Update start if same type and more extreme
            if start_fx.type == end_fx.type:
                if start_fx.type == "ding" and end_fx.val >= start_fx.val:
                    start_fx = end_fx
                elif start_fx.type == "di" and end_fx.val <= start_fx.val:
                    start_fx = end_fx
            
            idx += 1
            
        if not self.bis:
            return
            
        # 2. Continue from the end of the first pen
        curr_fx_idx = self.bis[-1].end.index + 1
        
        while curr_fx_idx < len(self.fxs):
            # Always reference the end of the LAST pen in self.bis
            # This allows us to modify the last pen (Merge)
            last_bi = self.bis[-1]
            start_fx = last_bi.end
            next_fx = self.fxs[curr_fx_idx]
            
            # Check 1: Same Type Extension (Merge Logic)
            # If next_fx is same type as start_fx (end of last pen)
            # And it is more extreme (Higher Top or Lower Bottom)
            # Then the previous pen ended prematurely. Extend it.
            if next_fx.type == start_fx.type:
                updated = False
                if start_fx.type == "ding" and next_fx.val >= start_fx.val:
                    updated = True
                elif start_fx.type == "di" and next_fx.val <= start_fx.val:
                    updated = True
                    
                if updated:
                    # Extend the last pen
                    last_bi.end = next_fx
                    last_bi.high, last_bi.low = self._bi_high_low(last_bi.start, next_fx)
                    # 笔被延伸，需要重新验证特征序列
                    last_bi.feature_sequence_verified = self._verify_feature_sequence(last_bi)
                    # Note: We do not add a new pen. We modified the existing one.
                
                curr_fx_idx += 1
                continue

            # Check 2: Different Type (Potential New Pen)
            if self.check_bi_valid(start_fx, next_fx, bi_type):
                # 增加下一笔确认逻辑
                # 笔必须要等下一个反向笔确认，如果相反分型K线个数不满足，笔会延伸
                is_confirm = False
                # 如果是最后一个分型，直接确认（或者是未完成的笔）
                if curr_fx_idx == len(self.fxs) - 1:
                    is_confirm = True
                else:
                    # 向后查找，看能否形成下一笔
                    check_idx = curr_fx_idx + 1
                    while check_idx < len(self.fxs):
                        check_fx = self.fxs[check_idx]
                        # 1. 如果遇到同向分型（与 next_fx 同向），且更极端，说明 next_fx 会延伸，当前 next_fx 无效
                        if check_fx.type == next_fx.type:
                            if next_fx.type == "ding" and check_fx.val >= next_fx.val:
                                break # next_fx 被延伸，当前不成笔
                            if next_fx.type == "di" and check_fx.val <= next_fx.val:
                                break # next_fx 被延伸，当前不成笔
                        
                        # 2. 检查能否成笔 (优先检查反向笔确认)
                        # 如果能形成有效的下一笔，则当前笔得到确认
                        if self.check_bi_valid(next_fx, check_fx, bi_type):
                            is_confirm = True
                            break

                        # 3. 如果遇到反向分型（与 next_fx 反向，即与 start_fx 同向），且更极端
                        # 并且此时还没有形成有效的下一笔（否则上面就break了）
                        # 说明 start_fx 会延伸，当前笔无效
                        if check_fx.type == start_fx.type:
                             if start_fx.type == "ding" and check_fx.val >= start_fx.val:
                                 break # start_fx 被延伸
                             if start_fx.type == "di" and check_fx.val <= start_fx.val:
                                 break # start_fx 被延伸
                        
                        # 4. 处理方向改变的毛刺（Top -> Bottom -> Top -> Top）
                        # 如果 check_fx 与 start_fx 同向（即与 next_fx 反向），但没有更极端
                        # 且它也不能与 next_fx 成笔（条件2已检查）
                        # 这说明 next_fx -> check_fx 是一个无效的波动（毛刺）
                        # 我们需要继续往后看，看是否有更极端的点来延伸 start_fx，或者有新的 valid 结构
                        # 这里不需要 break，继续循环即可。
                        
                        # 但有一个特殊情况：如果 check_fx 虽然没更极端，但后续紧接着一个同向点更极端了呢？
                        # 例如：Start(Top) -> Next(Bottom, Valid) -> Check1(Top, Invalid) -> Check2(Top, > Start)
                        # 这种情况下，Next 应该是无效的，Start 应该直接连到 Check2。
                        # 这里的逻辑已经在 Case 3 中涵盖了：只要 Check2 > Start，就会 break 并判定 Start 被延伸。
                        # 所以关键是：在遇到 Check2 之前，不要误判 Next 是有效的。
                        # 当前逻辑：如果没有遇到 valid next pen，循环会继续，直到遇到 Case 3 (break, not confirm) 或 Case 2 (confirm)
                        
                        check_idx += 1
                    
                    # 如果遍历完都没找到确认笔，且没有被延伸，说明后续震荡收敛或数据结束
                    # 如果是因为数据结束（循环正常结束），可以算作确认（未完成）
                    if not is_confirm and check_idx >= len(self.fxs):
                        is_confirm = True

                if is_confirm:
                    new_bi = BI(
                        start=start_fx,
                        end=next_fx,
                        _type="down" if start_fx.type == "ding" else "up",
                        index=len(self.bis)
                    )
                    new_bi.high, new_bi.low = self._bi_high_low(start_fx, next_fx)
                    
                    # 验证特征序列分型
                    new_bi.feature_sequence_verified = self._verify_feature_sequence(new_bi)
                    
                    # 检查是否可以确认之前的笔
                    if len(self.pending_bis) > 0:
                        last_pending_bi = self.pending_bis[-1]
                        # 通过反向下一笔来确认前一笔
                        if self._confirm_bi_with_next_bi(last_pending_bi, new_bi):
                            last_pending_bi.confirmed = True
                            # 从待确认列表中移除已确认的笔
                            self.pending_bis.pop()
                    
                    # 将新笔加入待确认列表
                    self.pending_bis.append(new_bi)
                    self.bis.append(new_bi)
                else:
                    # Not confirmed, ignore next_fx (treat as glitch)
                    pass
            else:
                # Invalid New Pen.
                # Just ignore next_fx.
                pass
                
            curr_fx_idx += 1

    def _cal_xd(self):
        self.xds = []
        if len(self.bis) < 3:
            return

        # 配置：是否允许笔破坏
        allow_bi_pohuai = self.config.get("xd_allow_bi_pohuai", "yes")

        start_idx = 0

        # 0. 初始线段方向智能识别
        # 如果起始阶段处于明显的趋势中，但第一笔的方向与趋势相反，会导致第一段线段无法结束（找不到特征序列的分型），从而产生超长线段
        # 因此，这里预读前几笔，判断局部趋势，强制第一笔顺应趋势
        if len(self.bis) > 4:
            up_score = 0
            down_score = 0
            # 检查前10笔的趋势特征
            check_len = min(12, len(self.bis))
            for i in range(2, check_len):
                # 仅比较同向笔
                if self.bis[i].type == self.bis[i-2].type:
                    if self.bis[i].high > self.bis[i-2].high:
                        up_score += 1
                    if self.bis[i].low > self.bis[i-2].low:
                        up_score += 1
                    if self.bis[i].high < self.bis[i-2].high:
                        down_score += 1
                    if self.bis[i].low < self.bis[i-2].low:
                        down_score += 1
            
            # 如果趋势明显，调整起始位置
            # 阈值设为 check_len // 3，确保有一定的趋势性
            threshold = check_len // 3
            if up_score > down_score + threshold: # 显著上涨
                if self.bis[start_idx].type == "down":
                    start_idx += 1
            elif down_score > up_score + threshold: # 显著下跌
                if self.bis[start_idx].type == "up":
                    start_idx += 1

        while start_idx <= len(self.bis) - 3:
            start_bi = self.bis[start_idx]
            
            # 1. 确定目标线段方向
            xd_dir = start_bi.type
            if len(self.xds) > 0:
                last_xd = self.xds[-1]
                # 必须与上一线段反向
                required_dir = "down" if last_xd.type == "up" else "up"
                
                if start_bi.type != required_dir:
                    # 方向相同，检查是否延伸上一线段
                    is_extension = False
                    if last_xd.type == "up":
                        if start_bi.high > last_xd.high:
                            is_extension = True
                            last_xd.high = start_bi.high
                            last_xd.end = start_bi.end
                            last_xd.end_line = start_bi
                    else:
                        if start_bi.low < last_xd.low:
                            is_extension = True
                            last_xd.low = start_bi.low
                            last_xd.end = start_bi.end
                            last_xd.end_line = start_bi
                    
                    if is_extension:
                        # 延伸了，上一线段结束点变更为当前笔，继续往后找
                        start_idx = start_bi.index + 1
                        continue
                    else:
                        # 同向但未延伸（包含在内部），跳过该笔，寻找下一个反向笔
                        start_idx += 1
                        continue
                
                xd_dir = required_dir

            # 第一段线段的特殊检查
            # if len(self.xds) == 0:
            #      bi1, bi2, bi3 = self.bis[start_idx], self.bis[start_idx+1], self.bis[start_idx+2]
            #      # 简单的重叠检查，确保不是单边上涨/下跌中途开始
            #      if not (max(bi1.low, bi2.low, bi3.low) < min(bi1.high, bi2.high, bi3.high)):
            #          start_idx += 1
            #          continue

            # 2. 特征序列分析与线段构建
            feature_dir = "down" if xd_dir == "up" else "up"
            
            # 严格检查：如果当前笔方向与目标线段方向不一致（同向），说明逻辑有误，跳过
            if start_bi.type != xd_dir:
                start_idx += 1
                continue

            tzxls: List[TZXL] = []
            made = False
            
            # 笔破坏检查标记
            # 如果刚开始就遇到反向笔破坏（V反），直接生成线段
            
            current_search_idx = start_idx + 1
            while current_search_idx < len(self.bis):
                bi = self.bis[current_search_idx]
                
                # 笔破坏判断 (仅在特征序列为空时检查，即线段刚开始的第一笔反向)
                # 修正：为了避免震荡中频繁出现微小线段破坏导致线段切碎，这里增加限制
                # 只有当反向笔力度极大（例如有缺口，或者跌幅巨大）时才允许笔破坏
                # 暂时策略：仅当配置明确允许且满足缺口条件时触发？或者直接从严，取消普通笔破坏。
                # 根据用户反馈，历史数据容易乱，建议从严。
                # 这里改为：必须有缺口才能触发笔破坏（即 V反+跳空）
                
                if allow_bi_pohuai == "yes" and len(tzxls) == 0 and bi.type == feature_dir:
                    pohuai = False
                    has_gap = False
                    
                    if xd_dir == "up":
                        if bi.low < start_bi.low:
                            pohuai = True
                            # 检查是否有缺口 (反向笔的高点 低于 前一笔的低点？不对，是反向笔的最高点 < start_bi 的最低点)
                            # start_bi 是 Up 笔。 bi 是 Down 笔。
                            # 缺口：Down笔最高点 < Up笔最低点 (直接跳空低开低走)
                            if bi.high < start_bi.low: has_gap = True
                    elif xd_dir == "down":
                        if bi.high > start_bi.high:
                            pohuai = True
                            # 缺口：Up笔最低点 > Down笔最高点
                            if bi.low > start_bi.high: has_gap = True
                    
                    # 只有在有缺口 或 破坏幅度极大（怎么定义极大？暂时只用缺口）的情况下才触发
                    # 或者，为了解决用户的“断掉”问题，我们暂时屏蔽掉无缺口的笔破坏
                    if pohuai and has_gap:
                        # 发生笔破坏，线段直接结束
                        xd = XD(start_bi.start, bi.end, start_bi, bi, xd_dir, None, None, len(self.xds))
                        if xd_dir == "up":
                            xd.high = start_bi.high
                            xd.low = bi.low
                        else:
                            xd.high = bi.high
                            xd.low = start_bi.low
                        xd.done = True
                        xd.is_split = "bi_pohuai"
                        self.xds.append(xd)
                        
                        # 新线段从破坏笔开始（V反）
                        start_idx = bi.index
                        made = True
                        break

                # 收集特征序列
                if bi.type == feature_dir:
                    # 检查是否触发上一线段的延伸
                    # 如果当前特征序列笔（与线段方向相反的笔，即反向线段的内部同向笔）突破了上一线段的极值
                    # 说明上一线段并未结束，而是发生了延伸
                    is_ext = False
                    if len(self.xds) > 0:
                        last_xd = self.xds[-1]
                        if last_xd.type == "up" and feature_dir == "up":
                            if bi.high > last_xd.high:
                                last_xd.high = bi.high
                                last_xd.end = bi.end
                                last_xd.end_line = bi
                                is_ext = True
                        elif last_xd.type == "down" and feature_dir == "down":
                            if bi.low < last_xd.low:
                                last_xd.low = bi.low
                                last_xd.end = bi.end
                                last_xd.end_line = bi
                                is_ext = True
                    
                    if is_ext:
                        start_idx = bi.index + 1
                        made = True
                        break

                    tz = TZXL(feature_dir, bi, None, False, True)
                    tz.max = bi.high
                    tz.min = bi.low
                    tz.lines = [bi]

                    # 包含处理
                    if tzxls:
                        last = tzxls[-1]
                        inc = (tz.max <= last.max and tz.min >= last.min) or (last.max <= tz.max and last.min >= tz.min)
                        if inc:
                            # 包含处理方向取决于线段方向
                            if xd_dir == "up": # 向上线段，特征序列向上处理（高高）
                                last.max = max(last.max, tz.max)
                                last.min = max(last.min, tz.min)
                            else: # 向下线段，特征序列向下处理（低低）
                                last.max = min(last.max, tz.max)
                                last.min = min(last.min, tz.min)
                            last.lines.append(bi)
                        else:
                            tzxls.append(tz)
                    else:
                        tzxls.append(tz)

                    # 分型判断
                    if len(tzxls) >= 3:
                        t1, t2, t3 = tzxls[-3], tzxls[-2], tzxls[-1]
                        is_fx = False
                        
                        if xd_dir == "up":
                            # 向上线段，特征序列(下笔)找顶分型
                            if t2.max >= t1.max and t2.max >= t3.max:
                                is_fx = True
                        else:
                            # 向下线段，特征序列(上笔)找底分型
                            if t2.min <= t1.min and t2.min <= t3.min:
                                is_fx = True
                        
                        if is_fx:
                            # 缺口判断 (Check Gap between 1st and 3rd element)
                            # 特征序列缺口定义：分型的第一元素和第三元素之间没有重叠区间
                            has_gap = False
                            if xd_dir == "up":
                                # 向上线段，特征序列(下笔)找顶分型
                                # 第一元素(t1)与第三元素(t3)无重叠
                                # 既然是顶分型，t2高点最高。如果t3的最高点 < t1的最低点，则肯定无重叠（巨大跳空）
                                # 或者 t3的最低点 > t1的最高点？(不可能，因为t3是下笔，t1也是下笔，中间隔着t2顶)
                                # 通常缺口是指：t3 range is completely below t1 range.
                                # t1: [min, max], t3: [min, max]
                                # Gap exists if t3.max < t1.min
                                if t3.max < t1.min: has_gap = True
                            else:
                                # 向下线段，特征序列(上笔)找底分型
                                # 第一元素(t1)与第三元素(t3)无重叠
                                # 底分型，t2低点最低。
                                # Gap exists if t3.min > t1.max
                                if t3.min > t1.max: has_gap = True
                            
                            # 破坏确认 (Break Condition)
                            break_condition = False
                            if xd_dir == "up":
                                if t3.min < t1.min: break_condition = True
                            else:
                                if t3.max > t1.max: break_condition = True
                            
                            is_valid = False
                            if has_gap:
                                # 有缺口，当下成立 (Standard: Gap -> Valid immediately)
                                is_valid = True
                            else:
                                # 无缺口，需确认 (Standard: No Gap -> Need Confirmation)
                                # 这里的确认通常指：后续走势不能收回分型区间，或者t3力度足够大
                                # 简化处理：如果满足 break_condition (即 t3 突破了 t1 的极值)，则视为有效
                                if break_condition: is_valid = True
                            
                            if is_valid:
                                # 找到分型顶点
                                if xd_dir == "up":
                                    peak = max(t2.lines, key=lambda b: b.high)
                                    fx_obj = XLFX("ding", t2, [t1, t2, t3], True)
                                else:
                                    peak = min(t2.lines, key=lambda b: b.low)
                                    fx_obj = XLFX("di", t2, [t1, t2, t3], True)
                                fx_obj.qk = has_gap
                                fx_obj.is_line_bad = False

                                # 线段结束点：极值笔的前一笔
                                # peak 是特征序列的元素（反向笔）。
                                # 向上线段结束于 Peak(下笔) 的起始点。
                                # Peak 的起始点是前一笔(上笔) 的结束点。
                                end_bi_idx = peak.index - 1
                                if end_bi_idx < start_idx: continue # 异常保护
                                
                                end_bi = self.bis[end_bi_idx]
                                
                                xd = XD(
                                    start_bi.start,
                                    end_bi.end,
                                    start_bi,
                                    end_bi,
                                    xd_dir,
                                    fx_obj if xd_dir == "up" else None,
                                    fx_obj if xd_dir == "down" else None,
                                    len(self.xds)
                                )
                                if xd_dir == "up":
                                    xd.high = end_bi.end.val
                                    xd.low = start_bi.start.val
                                else:
                                    xd.high = start_bi.start.val
                                    xd.low = end_bi.end.val
                                xd.done = True
                                self.xds.append(xd)
                                
                                # 下一段从 Peak 笔开始
                                start_idx = peak.index
                                made = True
                                break
                
                current_search_idx += 1
            
            if made:
                continue
            
            # 如果遍历完所有笔都没生成线段
            # 尝试策略：
            # 1. 检查是否可以延伸上一线段（即假突破/震荡后延续原趋势）
            # 2. 如果无法延伸，再考虑强制分段（防止死循环）
            
            extension_success = False
            if len(self.xds) > 0:
                last_xd = self.xds[-1]
                # 向后搜索一定范围，看是否有创新高/新低的笔
                search_limit = min(len(self.bis), start_idx + 100)
                
                if last_xd.type == "up":
                    # 上涨线段，寻找更高点
                    for k in range(start_idx, search_limit):
                        # 必须是同向笔（Up笔）创新高才算有效延伸
                        if self.bis[k].type == "up" and self.bis[k].high > last_xd.high:
                            # 找到新高，说明之前的线段结束是误判（或者行情延续）
                            # 延伸上一线段
                            last_xd.high = self.bis[k].high
                            last_xd.end = self.bis[k].end
                            last_xd.end_line = self.bis[k]
                            
                            # 重置搜索起点为新高点的下一笔
                            start_idx = k + 1
                            extension_success = True
                            break
                else:
                    # 下跌线段，寻找更低点
                    for k in range(start_idx, search_limit):
                        # 必须是同向笔（Down笔）创新低才算有效延伸
                        if self.bis[k].type == "down" and self.bis[k].low < last_xd.low:
                            # 找到新低，延伸
                            last_xd.low = self.bis[k].low
                            last_xd.end = self.bis[k].end
                            last_xd.end_line = self.bis[k]
                            
                            start_idx = k + 1
                            extension_success = True
                            break
            
            if extension_success:
                continue

            # 1. 如果剩余笔数不多，可能是真的未完成，直接退出处理未完成逻辑
            if len(self.bis) - start_idx < 20:
                break
            
            # 2. 如果剩余笔数很多，且无法延伸，说明可能是复杂震荡
            # 为了防止后续线段全部丢失，进行“强制分段”容错处理
            # 策略：在后续的一段范围内（例如20笔），寻找当前方向的极值点，强制结束
            
            search_range = min(len(self.bis), start_idx + 30)
            candidates = self.bis[start_idx:search_range]
            
            if xd_dir == "up":
                # 向上线段，找最高点强制结束
                best_bi = max(candidates, key=lambda b: b.high)
            else:
                # 向下线段，找最低点强制结束
                best_bi = min(candidates, key=lambda b: b.low)
            
            # 强制生成线段
            # 注意：best_bi 必须是同向的。如果是反向的，取其前一笔（同向）
            if best_bi.type != xd_dir:
                # 这种情况很少见，因为 max/min 会找到极值。
                # 如果极值笔恰好是反向笔（例如向上线段中，一个反向笔的高点比同向笔还高？不可能，同向笔肯定连接更高的位置）
                # 除非是包含关系。
                # 简单起见，如果 best_bi 是反向，回退到前一笔
                if best_bi.index > start_idx:
                    best_bi = self.bis[best_bi.index - 1]
            
            # 确保 best_bi 在 start_bi 之后
            if best_bi.index <= start_idx:
                 # 实在找不到，强制前移，避免死循环
                 start_idx += 1
                 continue

            xd = XD(
                start_bi.start,
                best_bi.end,
                start_bi,
                best_bi,
                xd_dir,
                None,
                None,
                len(self.xds)
            )
            if xd_dir == "up":
                xd.high = best_bi.end.val
                xd.low = start_bi.start.val
            else:
                xd.high = start_bi.start.val
                xd.low = best_bi.end.val
            
            # 标记为强制分段
            xd.done = True
            xd.is_split = "force" 
            self.xds.append(xd)
            
            # 下一段从强制结束点的下一笔开始（反向）
            start_idx = best_bi.index + 1
            # 此时必须确保下一笔是反向的，理论上笔是交替的，所以 best_bi(同向) 的下一笔一定是反向
            continue
        
        # 处理未完成线段
        if len(self.xds) > 0:
            last_xd = self.xds[-1]
            # 从上一线段结束位置开始，到最后一笔
            # 注意 start_idx 已经被更新为上一线段结束位置的下一笔（即新段开始）
            if start_idx < len(self.bis):
                last_bi = self.bis[-1]
                
                # 确保方向正确
                # 此时 start_idx 对应的笔应该是反向的
                # 如果不是，说明已经在上面的循环中被跳过或合并了，这里 start_idx 指向的一定是反向笔或者列表末尾
                
                # 如果剩余笔数太少，或者没有构成反向趋势？
                # 无论如何，剩余部分作为未完成段
                
                xd_dir = "down" if last_xd.type == "up" else "up"
                # 检查 start_idx 的笔是否匹配方向，如果不匹配（理论上不会，因为上面的循环保证了 start_idx 指向反向笔或结束），
                # 但如果最后几笔都是同向的延伸，它们会在上面被合并。
                # 所以这里直接连接即可。
                
                start_bi = self.bis[start_idx]
                # 简单检查方向，如果不匹配，可能需要调整 start_idx (虽然逻辑上应该匹配)
                if start_bi.type != xd_dir:
                    # 这种情况可能是：最后几笔是同向的，但是没有创新高/新低（未合并），被跳过了。
                    # 比如 Up段结束，后面是 Down(不创新低), Up, Down...
                    # 应该找到第一个符合方向的笔？
                    # 或者直接取 start_idx？
                    # 既然是未完成，我们假设它延续到最后。
                    pass

                xd = XD(start_bi.start, last_bi.end, start_bi, last_bi, xd_dir, None, None, len(self.xds))
                if xd_dir == "up":
                    xd.high = max([b.high for b in self.bis[start_idx:]])
                    xd.low = start_bi.start.val
                else:
                    xd.high = start_bi.start.val
                    xd.low = min([b.low for b in self.bis[start_idx:]])
                xd.done = False
                self.xds.append(xd)
        else:
            # 一根线段都没生成，创建一个未完成的
            if len(self.bis) > 0:
                xd = XD(self.bis[0].start, self.bis[-1].end, self.bis[0], self.bis[-1], self.bis[0].type, None, None, 0)
                if xd.type == "up":
                    xd.high = max([b.high for b in self.bis])
                    xd.low = self.bis[0].start.val
                else:
                    xd.high = self.bis[0].start.val
                    xd.low = min([b.low for b in self.bis])
                xd.done = False
                self.xds.append(xd)


    def _cal_zsd(self):
        self.zsds = self.xds

    def _cal_zs(self):
        # Clear existing ZSs
        self.bi_zss = {}
        self.xd_zss = {}

        # Helper: 以三线重叠计算中枢，并向后延伸直到离开（含离开段）
        def _calc_zss_by_lines(_lines: List[LINE], _zs_type: str) -> List[ZS]:
            zss: List[ZS] = []
            if len(_lines) < 3:
                return zss
            i = 0
            while i <= len(_lines) - 3:
                l1, l2, l3 = _lines[i], _lines[i + 1], _lines[i + 2]
                zg = min(l1.high, l2.high, l3.high)
                zd = max(l1.low, l2.low, l3.low)
                if zg > zd:
                    zs = ZS(
                        zs_type=_zs_type,
                        start=l1.start,
                        end=l3.end,
                        zg=zg,
                        zd=zd,
                        gg=max(l1.high, l2.high, l3.high),
                        dd=min(l1.low, l2.low, l3.low),
                        index=len(zss),
                        line_num=3,
                        _type="up" if l1.type == "down" else "down",
                    )
                    zs.lines = [l1, l2, l3]
                    zs.real = True
                    j = i + 3
                    # 段内延伸（含离开段）
                    while j < len(_lines):
                        ln = _lines[j]
                        if not (ln.high < zd or ln.low > zg):
                            zs.lines.append(ln)
                            zs.end = ln.end
                            if ln.high > zs.gg:
                                zs.gg = ln.high
                            if ln.low < zs.dd:
                                zs.dd = ln.low
                            j += 1
                        else:
                            zs.lines.append(ln)  # 计入离开段以确定右边界
                            zs.end = ln.end
                            if ln.high > zs.gg:
                                zs.gg = ln.high
                            if ln.low < zs.dd:
                                zs.dd = ln.low
                            j += 1
                            break
                    zss.append(zs)
                    i = j - 1
                else:
                    i += 1
            return zss

        # --- 计算笔中枢（支持标准/段内） ---
        zs_bi_types = self.config.get("zs_bi_type", [])
        if not isinstance(zs_bi_types, list):
            zs_bi_types = [zs_bi_types]

        for zs_type in zs_bi_types:
            self.bi_zss[zs_type] = []
            if len(self.bis) < 3:
                continue

            if zs_type == Config.ZS_TYPE_DN.value and len(self.xds) > 0:
                # 段内中枢：每个线段内独立计算中枢，从线段起点开始重算
                for xd in self.xds:
                    start_idx = xd.start_line.index
                    end_idx = xd.end_line.index if xd.end_line is not None else self.bis[-1].index
                    sub_lines = [bi for bi in self.bis if start_idx <= bi.index <= end_idx]
                    if len(sub_lines) < 3:
                        continue
                    zss_dn = _calc_zss_by_lines(sub_lines, zs_type)
                    # 修正中枢索引为全局顺序
                    for zs in zss_dn:
                        zs.index = len(self.bi_zss[zs_type])
                        self.bi_zss[zs_type].append(zs)
            else:
                # 标准中枢/其他类型：在全体笔上计算
                self.bi_zss[zs_type] = _calc_zss_by_lines(self.bis, zs_type)

        # --- 计算线段中枢（支持标准/段内） ---
        zs_xd_types = self.config.get("zs_xd_type", [])
        if not isinstance(zs_xd_types, list):
            zs_xd_types = [zs_xd_types]

        for zs_xd_type in zs_xd_types:
            self.xd_zss[zs_xd_type] = []
            if len(self.xds) < 3:
                continue

            if zs_xd_type == Config.ZS_TYPE_DN.value:
                # 段内中枢：按走势段内线段进行局部计算（此处线段本身已是特征序列产物，直接全量计算即可）
                self.xd_zss[zs_xd_type] = _calc_zss_by_lines(self.xds, zs_xd_type)
            else:
                # 标准中枢/其他类型：同样按全量线段计算
                self.xd_zss[zs_xd_type] = _calc_zss_by_lines(self.xds, zs_xd_type)

    def _cal_mmd_bc(self):
        # Calculate MACD LD (Force) and Check BC (Divergence)
        # And Identify MMD (Buy/Sell Points)
        
        # --- 1. Calculate for Strokes (BI) ---
        zs_type = self.config.get("zs_bi_type", ["common"])[0]
        zss = self.bi_zss.get(zs_type, [])
        
        # Configuration flags for Buy/Sell Points
        check_1buy = self.config.get("mmd_1buy_bc", True)
        check_1sell = self.config.get("mmd_1sell_bc", True)
        check_2buy = self.config.get("mmd_2buy_bc", True)
        check_2sell = self.config.get("mmd_2sell_bc", True)
        check_3buy = self.config.get("mmd_3buy_bc", True)
        check_3sell = self.config.get("mmd_3sell_bc", True)
        check_l2buy = self.config.get("mmd_l2buy_bc", True)
        check_l2sell = self.config.get("mmd_l2sell_bc", True)
        check_l3buy = self.config.get("mmd_l3buy_bc", True)
        check_l3sell = self.config.get("mmd_l3sell_bc", True)

        for i in range(len(self.bis)):
            bi = self.bis[i]
            if i >= 2:
                _window = self.bis[: i + 1]
                try:
                    _hist_bc, _deadif_bc = cal_macd_bis_is_bc(_window, self)
                except Exception:
                    _hist_bc, _deadif_bc = (False, False)
                if _hist_bc or _deadif_bc:
                    bi.add_bc("bi", None, self.bis[i-2], [self.bis[i-2]], True, zs_type)
            
            # A. Calculate MACD LD
            # (Already handled by get_ld internally or we assume it's available)
            
            # B. Check Basic Divergence (Compare with previous same-direction stroke)
            if i >= 2:
                prev_bi = self.bis[i-2]
                if prev_bi.type == bi.type:
                    ld1 = bi.get_ld(self)["macd"]
                    ld2 = prev_bi.get_ld(self)["macd"]
                    
                    if compare_ld_beichi(ld2, ld1, bi.type):
                        # Check if it makes a new High/Low
                        if (bi.type == "up" and bi.high > prev_bi.high) or \
                           (bi.type == "down" and bi.low < prev_bi.low):
                             bi.add_bc("bi", None, prev_bi, [prev_bi], True, zs_type)

            # C. Check MMD (Buy/Sell Points) relative to Pivots
            
            # Check if bi is associated with any ZS logic
            for zs in zss:
                # Only consider pivots that are "finished" or relevant
                if not zs.real: continue
                
                # PZ (Consolidation Divergence)
                # Check if 'bi' is the Leaving Stroke of 'zs'
                is_pz, compare_line = self.beichi_pz(zs, bi)
                if is_pz:
                    bi.add_bc("pz", zs, compare_line, [compare_line], True, zs_type)
                
                # QS (Trend Divergence)
                # Check if 'bi' is the end of a trend relative to 'zs'
                is_qs, compare_lines = self.beichi_qs(self.bis, zss, bi)
                if is_qs:
                     # Ensure the divergence is related to THIS zs (last pivot of trend)
                     if zs.lines[-1].index < bi.index: # ZS must be before BI
                         bi.add_bc("qs", zs, None, compare_lines, True, zs_type)
                
                # 3rd Buy/Sell
                if i >= 1:
                    prev_bi = self.bis[i-1]
                    if prev_bi.start.index == zs.end.index:
                        # Check 3rd Buy
                        if check_3buy and bi.type == "down" and bi.low > zs.zg:
                             if not any(m.name == "3buy" and m.zs.index == zs.index for m in bi.mmds):
                                 bi.add_mmd("3buy", zs, zs_type)
                        
                        # Check 3rd Sell
                        if check_3sell and bi.type == "up" and bi.high < zs.zd:
                             if not any(m.name == "3sell" and m.zs.index == zs.index for m in bi.mmds):
                                 bi.add_mmd("3sell", zs, zs_type)
                
                # 1st Buy/Sell (Trend Divergence)
                has_prev_3sell = False
                has_prev_3buy = False
                if i >= 1:
                    prev_bi = self.bis[i-1]
                    if bi.type == "down":
                        has_prev_3sell = any(m.name == "3sell" for m in prev_bi.get_mmds(zs_type))
                    if bi.type == "up":
                        has_prev_3buy = any(m.name == "3buy" for m in prev_bi.get_mmds(zs_type))

                is_new_extreme = True
                if i >= 2:
                    prev_bi_2 = self.bis[i-2]
                    if bi.type == "down" and bi.low >= prev_bi_2.low:
                        is_new_extreme = False
                    if bi.type == "up" and bi.high <= prev_bi_2.high:
                        is_new_extreme = False

                if is_new_extreme:
                    has_qs_zs = any(b.type == "qs" and b.zs.index == zs.index for b in bi.get_bcs(zs_type))
                    
                    if has_qs_zs: 
                        if check_1buy and bi.type == "down" and bi.low < zs.zd:
                            bi.add_mmd("1buy", zs, zs_type)
                        if check_1sell and bi.type == "up" and bi.high > zs.zg:
                            bi.add_mmd("1sell", zs, zs_type)
                    
                    elif (has_prev_3sell or has_prev_3buy):
                         has_div_zs = any(b.type in ["qs", "pz"] and b.zs.index == zs.index for b in bi.get_bcs(zs_type))
                         has_div_bi = bi.bc_exists(["bi"], zs_type)
                         
                         if has_div_zs or has_div_bi:
                             if check_1buy and bi.type == "down" and (bi.low < zs.zd or has_prev_3sell):
                                bi.add_mmd("1buy", zs, zs_type)
                             if check_1sell and bi.type == "up" and (bi.high > zs.zg or has_prev_3buy):
                                bi.add_mmd("1sell", zs, zs_type)

                # 2nd Buy/Sell
                if i >= 3:
                    prev_bi_2 = self.bis[i-2]
                    prev_bi_3 = self.bis[i-3]
                    
                    is_1buy = any(m.name == "1buy" for m in prev_bi_2.get_mmds(zs_type))
                    is_1sell = any(m.name == "1sell" for m in prev_bi_2.get_mmds(zs_type))

                    if check_2buy and bi.type == "down":
                        has_3sell_prev = any(m.name == "3sell" and m.zs.index == zs.index for m in prev_bi_3.get_mmds(zs_type))
                        if has_3sell_prev and not is_1buy and bi.low > prev_bi_2.low:
                            if not any(m.name == "2buy" and m.zs.index == zs.index for m in bi.get_mmds(zs_type)):
                                bi.add_mmd("2buy", zs, zs_type)

                    if check_2sell and bi.type == "up":
                        has_3buy_prev = any(m.name == "3buy" and m.zs.index == zs.index for m in prev_bi_3.get_mmds(zs_type))
                        if has_3buy_prev and not is_1sell and bi.high < prev_bi_2.high:
                            if not any(m.name == "2sell" and m.zs.index == zs.index for m in bi.get_mmds(zs_type)):
                                bi.add_mmd("2sell", zs, zs_type)

            # 2nd Buy/Sell (Cases independent of current ZS loop)
            if i >= 2:
                prev_bi_2 = self.bis[i-2]
                is_1buy = any(m.name == "1buy" for m in prev_bi_2.get_mmds(zs_type))
                is_1sell = any(m.name == "1sell" for m in prev_bi_2.get_mmds(zs_type))

                if check_2buy and bi.type == "down":
                    # Case 1 & 3: After 1buy
                    if is_1buy and bi.low > prev_bi_2.low:
                         target_zs = prev_bi_2.get_mmds(zs_type)[0].zs
                         if not any(m.name == "2buy" and m.zs.index == target_zs.index for m in bi.get_mmds(zs_type)):
                             bi.add_mmd("2buy", target_zs, zs_type)
                    
                    # Case 2: Trend + No New Low + Divergence
                    for b in prev_bi_2.get_bcs(zs_type):
                        if b.type in ["qs", "pz"] and b.zs is not None:
                             if bi.low > prev_bi_2.low and compare_ld_beichi(prev_bi_2.get_ld(self), bi.get_ld(self), "down"):
                                 if not any(m.name == "2buy" and m.zs.index == b.zs.index for m in bi.get_mmds(zs_type)):
                                     bi.add_mmd("2buy", b.zs, zs_type)
                                     break

                if check_2sell and bi.type == "up":
                    # Case 1 & 3
                    if is_1sell and bi.high < prev_bi_2.high:
                         target_zs = prev_bi_2.get_mmds(zs_type)[0].zs
                         if not any(m.name == "2sell" and m.zs.index == target_zs.index for m in bi.get_mmds(zs_type)):
                             bi.add_mmd("2sell", target_zs, zs_type)
                    
                    # Case 2
                    for b in prev_bi_2.get_bcs(zs_type):
                        if b.type in ["qs", "pz"] and b.zs is not None:
                             if bi.high < prev_bi_2.high and compare_ld_beichi(prev_bi_2.get_ld(self), bi.get_ld(self), "up"):
                                 if not any(m.name == "2sell" and m.zs.index == b.zs.index for m in bi.get_mmds(zs_type)):
                                     bi.add_mmd("2sell", b.zs, zs_type)
                                     break

            # Class 2/3 Buy/Sell (Independent of ZS loop)
            if i >= 2:
                prev_bi_2 = self.bis[i-2]
                prev_bi = self.bis[i-1]
                if bi.type == "down":
                    # Class 2 Buy
                    if check_l2buy:
                        has_2buy_mmds = [m for m in prev_bi_2.get_mmds(zs_type) if m.name == "2buy"]
                        if has_2buy_mmds:
                            zg = min(prev_bi_2.high, prev_bi.high, bi.high)
                            if zg > bi.low and bi.low > prev_bi_2.low:
                                 target_zs = has_2buy_mmds[0].zs
                                 if not any(m.name == "l2buy" and m.zs.index == target_zs.index for m in bi.get_mmds(zs_type)):
                                     bi.add_mmd("l2buy", target_zs, zs_type)
                    
                    # Class 3 Buy
                    if check_l3buy:
                        has_3buy_mmds = [m for m in prev_bi_2.get_mmds(zs_type) if m.name == "3buy"]
                        if has_3buy_mmds:
                            zg = min(prev_bi_2.high, prev_bi.high, bi.high)
                            if zg > bi.low and bi.low > prev_bi_2.low:
                                 target_zs = has_3buy_mmds[0].zs
                                 if not any(m.name == "l3buy" and m.zs.index == target_zs.index for m in bi.get_mmds(zs_type)):
                                     bi.add_mmd("l3buy", target_zs, zs_type)

                if bi.type == "up":
                    # Class 2 Sell
                    if check_l2sell:
                        has_2sell_mmds = [m for m in prev_bi_2.get_mmds(zs_type) if m.name == "2sell"]
                        if has_2sell_mmds:
                            zd = max(prev_bi_2.low, prev_bi.low, bi.low)
                            if bi.high > zd and bi.high < prev_bi_2.high:
                                 target_zs = has_2sell_mmds[0].zs
                                 if not any(m.name == "l2sell" and m.zs.index == target_zs.index for m in bi.get_mmds(zs_type)):
                                     bi.add_mmd("l2sell", target_zs, zs_type)
                    
                    # Class 3 Sell
                    if check_l3sell:
                        has_3sell_mmds = [m for m in prev_bi_2.get_mmds(zs_type) if m.name == "3sell"]
                        if has_3sell_mmds:
                            zd = max(prev_bi_2.low, prev_bi.low, bi.low)
                            if bi.high > zd and bi.high < prev_bi_2.high:
                                 target_zs = has_3sell_mmds[0].zs
                                 if not any(m.name == "l3sell" and m.zs.index == target_zs.index for m in bi.get_mmds(zs_type)):
                                     bi.add_mmd("l3sell", target_zs, zs_type)

            for zs in zss:
                # Only consider pivots that are "finished" or relevant
                if not zs.real: continue
                
                # PZ (Consolidation Divergence)
                if bi.start.index >= zs.start.index and bi.end.index <= zs.end.index:
                    # Inside Pivot - usually no PZ unless it's the entering/leaving segment comparison?
                    # PZ is usually compared Entering vs Leaving.
                    pass
                
                # Check PZ (Leaving Stroke vs Entering Stroke/Pivot Internal)
                # If bi is the leaving stroke?
                # Or if bi is currently completing and might be PZ.
                # We use helper function.
                is_pz, compare_line = self.beichi_pz(zs, bi)
                if is_pz:
                     bi.add_bc("pz", zs, compare_line, [compare_line], True, zs_type)
                else:
                    # fallback: 离开段前一笔即中枢最后一线，比较进入段与当前段力度
                    if i >= 1 and self.bis[i-1].index == zs.lines[-1].index:
                        enter_idx = zs.lines[0].index - 1
                        if enter_idx >= 0:
                            entering_line = self.bis[enter_idx]
                            if compare_ld_beichi(entering_line.get_ld(self), bi.get_ld(self), bi.type):
                                bi.add_bc("pz", zs, entering_line, [entering_line], True, zs_type)

                # Check QS (Trend Divergence)
                # Needs at least 2 pivots.
                # We pass all zss, it will filter.
                is_qs, compare_lines = self.beichi_qs(self.bis, zss, bi)
                if is_qs:
                    bi.add_bc("qs", zs, None, compare_lines, True, zs_type)

                # 3rd Buy/Sell: Pivot Breakout + Pullback
                # Strict Check: The stroke BEFORE current bi must be the one that LEFT the pivot.
                if i >= 1:
                    prev_bi = self.bis[i-1]
                    # 离开段等于中枢最后一线
                    if prev_bi.index == zs.lines[-1].index:
                        # 3rd Buy
                        if bi.type == "down" and bi.low > zs.zg:
                             if not any(m.name == "3buy" and m.zs.index == zs.index for m in bi.mmds):
                                 bi.add_mmd("3buy", zs, zs_type)
                        # 3rd Sell
                        if bi.type == "up" and bi.high < zs.zd:
                             if not any(m.name == "3sell" and m.zs.index == zs.index for m in bi.mmds):
                                 bi.add_mmd("3sell", zs, zs_type)

                # 1st Buy/Sell (Trend Divergence)
                # 1. Two centers (same level) trend divergence 1st buy/sell
                # 2. After 3rd sell/buy, divergence 1st buy/sell
                
                # Check for 3rd Buy/Sell on previous stroke
                has_prev_3sell = False
                has_prev_3buy = False
                if i >= 1:
                    prev_bi = self.bis[i-1]
                    if bi.type == "down":
                        has_prev_3sell = any(m.name == "3sell" for m in prev_bi.get_mmds(zs_type))
                    if bi.type == "up":
                        has_prev_3buy = any(m.name == "3buy" for m in prev_bi.get_mmds(zs_type))

                # Strictly require QS (Trend Divergence) or PZ (Consolidation Divergence) for 1st Buy/Sell
                # Must be a New Low (Buy) or New High (Sell) relative to previous same-direction stroke
                is_new_extreme = True
                if i >= 2:
                    prev_bi_2 = self.bis[i-2]
                    if bi.type == "down" and bi.low >= prev_bi_2.low:
                        is_new_extreme = False
                    if bi.type == "up" and bi.high <= prev_bi_2.high:
                        is_new_extreme = False

                if is_new_extreme:
                    # Strict 1st Buy: Trend Divergence (QS)
                    # Check if QS exists FOR THIS ZS（当前或上一笔）
                    has_qs_zs = any(
                        b.type == "qs" and b.zs.index == zs.index for b in bi.get_bcs(zs_type)
                    ) or (
                        i >= 1
                        and any(
                            b.type == "qs" and b.zs.index == zs.index for b in self.bis[i - 1].get_bcs(zs_type)
                        )
                    )

                    if has_qs_zs:
                        if bi.type == "down" and bi.low < zs.zd:
                            bi.add_mmd("1buy", zs, zs_type)
                        if bi.type == "up" and bi.high > zs.zg:
                            bi.add_mmd("1sell", zs, zs_type)
                    # 1st Buy after 3rd Sell (with Divergence)
                    elif has_prev_3sell or has_prev_3buy:
                        # Check for any divergence (QS, PZ, BI) related to this ZS or generic（当前或上一笔）
                        has_div_zs = any(
                            b.type in ["qs", "pz"] and b.zs.index == zs.index for b in bi.get_bcs(zs_type)
                        ) or (
                            i >= 1
                            and any(
                                b.type in ["qs", "pz"] and b.zs.index == zs.index for b in self.bis[i - 1].get_bcs(zs_type)
                            )
                        )
                        # For BI divergence, it's not tied to ZS, so we check general existence
                        has_div_bi = bi.bc_exists(["bi"], zs_type)

                        if has_div_zs or has_div_bi:
                            if bi.type == "down" and (bi.low < zs.zd or has_prev_3sell):
                                bi.add_mmd("1buy", zs, zs_type)
                            if bi.type == "up" and (bi.high > zs.zg or has_prev_3buy):
                                bi.add_mmd("1sell", zs, zs_type)

                # 2nd Buy/Sell
                # Case 4: After 3S no 1B produced, subsequent no new low also appears 2B
                # This depends on 'zs' loop variable (the pivot of the 3S)
                if i >= 3:
                    prev_bi_2 = self.bis[i-2]
                    prev_bi_3 = self.bis[i-3]
                    
                    # Check if prev_bi_2 was 1buy
                    is_1buy = any(m.name == "1buy" for m in prev_bi_2.get_mmds(zs_type))

                    if bi.type == "down":
                        has_3sell_prev = any(m.name == "3sell" and m.zs.index == zs.index for m in prev_bi_3.get_mmds(zs_type))
                        if has_3sell_prev and not is_1buy and bi.low > prev_bi_2.low:
                            if not any(m.name == "2buy" and m.zs.index == zs.index for m in bi.get_mmds(zs_type)):
                                bi.add_mmd("2buy", zs, zs_type)

                    if bi.type == "up":
                        has_3buy_prev = any(m.name == "3buy" and m.zs.index == zs.index for m in prev_bi_3.get_mmds(zs_type))
                        if has_3buy_prev and not is_1sell and bi.high < prev_bi_2.high:
                            if not any(m.name == "2sell" and m.zs.index == zs.index for m in bi.get_mmds(zs_type)):
                                bi.add_mmd("2sell", zs, zs_type)


        
        # Call user custom MMD
        if len(self.bis) > 0:
            user_custom_mmd(self, self.bis[-1], self.bis, zs_type, zss)
            
        # --- 2. Calculate for Segments (XD) ---
        zs_xd_type = self.config.get("zs_xd_type", ["common"])[0]
        xd_zss = self.xd_zss.get(zs_xd_type, [])
        
        if len(self.xds) > 0:
             # Copy-paste logic for XDs or refactor. 
             # For simplicity and safety, we duplicate with object replacements.
             for i in range(len(self.xds)):
                xd = self.xds[i]
                
                # B. Check Basic Divergence
                if i >= 2:
                    prev_xd = self.xds[i-2]
                    if prev_xd.type == xd.type:
                        ld1 = xd.get_ld(self)["macd"]
                        ld2 = prev_xd.get_ld(self)["macd"]
                        if compare_ld_beichi(ld2, ld1, xd.type):
                            if (xd.type == "up" and xd.high > prev_xd.high) or \
                               (xd.type == "down" and xd.low < prev_xd.low):
                                 xd.add_bc("xd", None, prev_xd, [prev_xd], True, zs_xd_type)

                if not xd_zss: continue
                
                # 2nd Buy/Sell (Independent of current ZS loop)
                if i >= 2:
                    prev_xd_2 = self.xds[i-2]
                    is_1buy = any(m.name == "1buy" for m in prev_xd_2.get_mmds(zs_xd_type))
                    is_1sell = any(m.name == "1sell" for m in prev_xd_2.get_mmds(zs_xd_type))

                    if xd.type == "down":
                        # Case 1 & 3
                        if is_1buy and xd.low > prev_xd_2.low:
                             target_zs = prev_xd_2.get_mmds(zs_xd_type)[0].zs
                             if not any(m.name == "2buy" and m.zs.index == target_zs.index for m in xd.get_mmds(zs_xd_type)):
                                 xd.add_mmd("2buy", target_zs, zs_xd_type)
                        
                        # Case 2: Trend + No New Low + Divergence
                        for b in prev_xd_2.get_bcs(zs_xd_type):
                            if b.type in ["qs", "pz"] and b.zs is not None:
                                 if xd.low > prev_xd_2.low and compare_ld_beichi(prev_xd_2.get_ld(self), xd.get_ld(self), "down"):
                                     if not any(m.name == "2buy" and m.zs.index == b.zs.index for m in xd.get_mmds(zs_xd_type)):
                                         xd.add_mmd("2buy", b.zs, zs_xd_type)
                                         break

                    if xd.type == "up":
                        # Case 1 & 3
                        if is_1sell and xd.high < prev_xd_2.high:
                             target_zs = prev_xd_2.get_mmds(zs_xd_type)[0].zs
                             if not any(m.name == "2sell" and m.zs.index == target_zs.index for m in xd.get_mmds(zs_xd_type)):
                                 xd.add_mmd("2sell", target_zs, zs_xd_type)

                        # Case 2
                        for b in prev_xd_2.get_bcs(zs_xd_type):
                            if b.type in ["qs", "pz"] and b.zs is not None:
                                 if xd.high < prev_xd_2.high and compare_ld_beichi(prev_xd_2.get_ld(self), xd.get_ld(self), "up"):
                                     if not any(m.name == "2sell" and m.zs.index == b.zs.index for m in xd.get_mmds(zs_xd_type)):
                                         xd.add_mmd("2sell", b.zs, zs_xd_type)
                                         break

                for zs in xd_zss:
                    if not zs.real: continue
                    
                    # PZ
                    is_pz, compare_line = self.beichi_pz(zs, xd)
                    if is_pz: xd.add_bc("pz", zs, compare_line, [compare_line], True, zs_xd_type)
                    
                    # QS
                    is_qs, compare_lines = self.beichi_qs(self.xds, xd_zss, xd)
                    if is_qs: xd.add_bc("qs", zs, None, compare_lines, True, zs_xd_type)
                    
                    # 3rd Buy/Sell
                    if i >= 1:
                        prev_xd = self.xds[i-1]
                        if prev_xd.start.index == zs.end.index:
                            if xd.type == "down" and xd.low > zs.zg:
                                 if not any(m.name == "3buy" and m.zs.index == zs.index for m in xd.mmds):
                                     xd.add_mmd("3buy", zs, zs_xd_type)
                            if xd.type == "up" and xd.high < zs.zd:
                                 if not any(m.name == "3sell" and m.zs.index == zs.index for m in xd.mmds):
                                     xd.add_mmd("3sell", zs, zs_xd_type)
                    
                    # 1st Buy/Sell
                    has_prev_3sell = False
                    has_prev_3buy = False
                    if i >= 1:
                        prev_xd = self.xds[i-1]
                        if xd.type == "down":
                            has_prev_3sell = any(m.name == "3sell" for m in prev_xd.get_mmds(zs_xd_type))
                        if xd.type == "up":
                            has_prev_3buy = any(m.name == "3buy" for m in prev_xd.get_mmds(zs_xd_type))

                    # Strict 1st Buy: Trend Divergence (QS)
                    has_qs_zs = any(b.type == "qs" and b.zs.index == zs.index for b in xd.get_bcs(zs_xd_type))
                    if has_qs_zs:
                        if xd.type == "down" and xd.low < zs.zd: xd.add_mmd("1buy", zs, zs_xd_type)
                        if xd.type == "up" and xd.high > zs.zg: xd.add_mmd("1sell", zs, zs_xd_type)
                    # 1st Buy after 3rd Sell (with Divergence)
                    elif (has_prev_3sell or has_prev_3buy):
                        has_div_zs = any(b.type in ["qs", "pz"] and b.zs.index == zs.index for b in xd.get_bcs(zs_xd_type))
                        has_div_xd = xd.bc_exists(["xd"], zs_xd_type)
                        
                        if has_div_zs or has_div_xd:
                             if xd.type == "down":
                                  if xd.low < zs.zd or has_prev_3sell: xd.add_mmd("1buy", zs, zs_xd_type)
                             if xd.type == "up":
                                  if xd.high > zs.zg or has_prev_3buy: xd.add_mmd("1sell", zs, zs_xd_type)
                        
                    # 2nd Buy/Sell
                    # Case 4: After 3S
                    if i >= 3:
                        prev_xd_2 = self.xds[i-2]
                        prev_xd_3 = self.xds[i-3]
                        is_1buy = any(m.name == "1buy" for m in prev_xd_2.get_mmds(zs_xd_type))
                        
                        if xd.type == "down":
                            has_3sell_prev = any(m.name == "3sell" and m.zs.index == zs.index for m in prev_xd_3.get_mmds(zs_xd_type))
                            if has_3sell_prev and not is_1buy and xd.low > prev_xd_2.low:
                                if not any(m.name == "2buy" and m.zs.index == zs.index for m in xd.get_mmds(zs_xd_type)):
                                    xd.add_mmd("2buy", zs, zs_xd_type)

                        if xd.type == "up":
                            has_3buy_prev = any(m.name == "3buy" and m.zs.index == zs.index for m in prev_xd_3.get_mmds(zs_xd_type))
                            if has_3buy_prev and not any(m.name == "1sell" for m in prev_xd_2.get_mmds(zs_xd_type)) and xd.high < prev_xd_2.high:
                                if not any(m.name == "2sell" and m.zs.index == zs.index for m in xd.get_mmds(zs_xd_type)):
                                    xd.add_mmd("2sell", zs, zs_xd_type)
                            


    def create_dn_zs(self, zs_type: str, lines: List[LINE], max_line_num: int = 999, zs_include_last_line=True) -> List[ZS]:
        zss = []
        if len(lines) < 3:
            return zss
            
        i = 0
        while i <= len(lines) - 3:
            l1 = lines[i]
            l2 = lines[i+1]
            l3 = lines[i+2]
            
            zg = min(l1.high, l2.high, l3.high)
            zd = max(l1.low, l2.low, l3.low)
            
            if zg > zd:
                zs = ZS(
                    zs_type=zs_type,
                    start=l1.start,
                    end=l3.end,
                    zg=zg,
                    zd=zd,
                    gg=max(l1.high, l2.high, l3.high),
                    dd=min(l1.low, l2.low, l3.low),
                    index=len(zss),
                    line_num=3,
                    _type="up" if l1.type == "down" else "down"
                )
                zs.lines = [l1, l2, l3]
                zs.real = True
                zss.append(zs)
                
                j = i + 3
                while j < len(lines):
                    ln = lines[j]
                    if not zs_include_last_line and j == len(lines) - 1:
                        break

                    if not (ln.high < zd or ln.low > zg):
                        if len(zs.lines) >= max_line_num:
                            break
                        zs.lines.append(ln)
                        zs.end = ln.end
                        if ln.high > zs.gg: zs.gg = ln.high
                        if ln.low < zs.dd: zs.dd = ln.low
                        j += 1
                    else:
                        break
                i = j - 1
            else:
                i += 1
        return zss

    def beichi_pz(self, zs: ZS, now_line: LINE) -> Tuple[bool, Union[LINE, None]]:
        if len(zs.lines) < 1:
            return False, None

        lines = self.bis if isinstance(now_line, BI) else (self.xds if isinstance(now_line, XD) else None)
        if lines is None:
            return False, None

        first = zs.lines[0]
        entering_idx = first.index - 1
        if entering_idx < 0:
            return False, None
        entering = lines[entering_idx]

        if now_line.start.index != zs.end.index:
            return False, None

        group = [entering] + zs.lines + [now_line]
        if now_line.type == "up":
            enter_extreme = entering.high == max(l.high for l in group)
            leave_extreme = now_line.high == max(l.high for l in group)
        else:
            enter_extreme = entering.low == min(l.low for l in group)
            leave_extreme = now_line.low == min(l.low for l in group)

        if not (enter_extreme and leave_extreme):
            return False, None

        ld1 = entering.get_ld(self)
        ld2 = now_line.get_ld(self)
        if compare_ld_beichi(ld1, ld2, now_line.type):
            return True, entering
        return False, None

    def beichi_qs(self, lines: List[LINE], zss: List[ZS], now_line: LINE) -> Tuple[bool, List[LINE]]:
        if len(zss) < 2:
            return False, []

        zs1 = zss[-2]
        zs2 = zss[-1]
        trend_type, _ = self.zss_is_qs(zs1, zs2)
        if not trend_type:
            return False, []

        if trend_type == "up" and now_line.type != "up":
            return False, []
        if trend_type == "down" and now_line.type != "down":
            return False, []

        prev_same = [l for l in lines if l.index > zs1.lines[-1].index and l.index < now_line.index and l.type == now_line.type]
        if not prev_same:
            return False, []
        ref = prev_same[-1]

        if now_line.type == "up" and not (now_line.high > ref.high):
            return False, []
        if now_line.type == "down" and not (now_line.low < ref.low):
            return False, []

        ld1 = ref.get_ld(self)
        ld2 = now_line.get_ld(self)
        if compare_ld_beichi(ld1, ld2, now_line.type):
            return True, [ref]
        return False, []

    def zss_is_qs(self, one_zs: ZS, two_zs: ZS) -> Tuple[str, None]:
        wzgx = self.config.get("zs_wzgx", Config.ZS_WZGX_ZGD.value)
        
        # Up Trend
        is_up = False
        if wzgx == Config.ZS_WZGX_ZGD.value:
            if two_zs.zd > one_zs.zg: is_up = True
        elif wzgx == Config.ZS_WZGX_ZGGDD.value:
            if two_zs.zd > one_zs.gg: is_up = True
        elif wzgx == Config.ZS_WZGX_GD.value:
            if two_zs.dd > one_zs.gg: is_up = True
            
        if is_up:
            return "up", None
            
        # Down Trend
        is_down = False
        if wzgx == Config.ZS_WZGX_ZGD.value:
            if two_zs.zg < one_zs.zd: is_down = True
        elif wzgx == Config.ZS_WZGX_ZGGDD.value:
            if two_zs.zg < one_zs.dd: is_down = True
        elif wzgx == Config.ZS_WZGX_GD.value:
            if two_zs.gg < one_zs.dd: is_down = True
            
        if is_down:
            return "down", None
            
        return None, None


def create_cl(code: str, frequency: str, config: Union[dict, None] = None) -> CL:
    return CL(code, frequency, config)
