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
            if (k.h >= last_ck.h and k.l <= last_ck.l) or (last_ck.h >= k.h and last_ck.l <= k.l):
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
        if start_fx.type == "ding" and end_fx.val >= start_fx.val:
            return False
        if start_fx.type == "di" and end_fx.val <= start_fx.val:
            return False

        # 缠论K线中心索引差值（用于判断是否存在独立缠论K线）
        ck_diff = end_fx.k.index - start_fx.k.index

        # 原始K线数量（用于新笔/简单笔判断）
        src_k_num = 0
        for ck in self.cl_klines[start_fx.k.index : end_fx.k.index + 1]:
            src_k_num += len(ck.klines)

        if bi_type == Config.BI_TYPE_OLD.value:
            # 老笔：分型不共用缠论K线，分型之间至少有一根独立缠论K线
            return ck_diff >= 2
        elif bi_type == Config.BI_TYPE_NEW.value:
            # 新笔：分型之间至少5根原始K线，且不共用缠论K线（至少一根独立缠论K线）
            return src_k_num >= 5 and ck_diff >= 2
        elif bi_type == Config.BI_TYPE_JDB.value:
            # 简单笔：至少5根原始K线即可
            return src_k_num >= 5
        elif bi_type == Config.BI_TYPE_DD.value:
            # 顶底成笔：出现相邻顶底即可
            return True
        else:
            return ck_diff >= 2

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
            is_included = (k.h >= last.h and k.l <= last.l) or (last.h >= k.h and last.l <= k.l)
            
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

    def _cal_bi(self):
        self.bis = []
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
                    # Note: We do not add a new pen. We modified the existing one.
                
                curr_fx_idx += 1
                continue

            # Check 2: Different Type (Potential New Pen)
            if self.check_bi_valid(start_fx, next_fx, bi_type):
                bi = BI(
                    start=start_fx,
                    end=next_fx,
                    _type="down" if start_fx.type == "ding" else "up",
                    index=len(self.bis)
                )
                bi.high, bi.low = self._bi_high_low(start_fx, next_fx)
                self.bis.append(bi)
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
        while start_idx <= len(self.bis) - 3:
            # 1. 检查起始三笔是否有重叠 (线段定义的必要条件)
            bi1, bi2, bi3 = self.bis[start_idx], self.bis[start_idx+1], self.bis[start_idx+2]
            if not (max(bi1.low, bi2.low, bi3.low) < min(bi1.high, bi2.high, bi3.high)):
                start_idx += 1
                continue

            start_bi = self.bis[start_idx]
            xd_dir = start_bi.type
            feature_dir = "down" if xd_dir == "up" else "up"
            tzxls: List[TZXL] = []
            made = False
            last_same = start_bi

            # 笔破坏检查
            bi_pohuai_done = False

            for i in range(start_idx + 1, len(self.bis)):
                bi = self.bis[i]
                
                # 笔破坏判断 (标准线段被第一笔反向笔直接破坏)
                if allow_bi_pohuai == "yes" and not bi_pohuai_done:
                    # 必须是特征序列方向的笔 (反向笔)
                    if bi.type == feature_dir:
                        pohuai = False
                        if xd_dir == "up" and bi.low < start_bi.low:
                            pohuai = True
                        elif xd_dir == "down" and bi.high > start_bi.high:
                            pohuai = True
                        
                        if pohuai:
                            # 发生笔破坏，线段直接结束
                            xd = XD(start_bi.start, bi.end, start_bi, bi, xd_dir, None, None, len(self.xds))
                            if xd_dir == "up":
                                xd.high = start_bi.high # 笔破坏通常只有第一笔
                                xd.low = bi.low
                            else:
                                xd.high = bi.high
                                xd.low = start_bi.low
                            xd.done = True
                            # 标记为笔破坏
                            xd.is_split = "bi_pohuai"
                            self.xds.append(xd)
                            
                            made = True
                            start_idx = i 
                            bi_pohuai_done = True # 标记已处理
                            break

                if bi.type == xd_dir:
                    last_same = bi
                if bi.type != feature_dir:
                    continue

                # 特征序列元素
                tz = TZXL(feature_dir, bi, None, False, True)
                tz.max = bi.high
                tz.min = bi.low
                tz.lines = [bi]

                # 特征序列包含处理
                if tzxls:
                    last = tzxls[-1]
                    # 包含关系判断：tz 在 last 中，或者 last 在 tz 中
                    inc = (tz.max <= last.max and tz.min >= last.min) or (last.max <= tz.max and last.min >= tz.min)
                    if inc:
                        # 包含处理方向：取决于线段方向
                        # 向上线段 (xd_dir='up')，特征序列是向下的笔，但包含处理方向与线段方向相同（向上处理，取高高）
                        if xd_dir == "up":
                            last.max = max(last.max, tz.max)
                            last.min = max(last.min, tz.min)
                        else:
                            # 向下线段：特征序列向上笔，但包含处理方向与线段方向相同（向下处理，取低低）
                            last.max = min(last.max, tz.max)
                            last.min = min(last.min, tz.min)
                        last.lines.append(bi)
                    else:
                        tzxls.append(tz)
                else:
                    tzxls.append(tz)

                # 检查特征序列分型
                if len(tzxls) >= 3:
                    t1, t2, t3 = tzxls[-3], tzxls[-2], tzxls[-1]
                    
                    # 分型判断
                    is_fx = False
                    if xd_dir == "up":
                        # 向上线段，找特征序列的顶分型 (因为特征序列是向下笔，趋势是上升的? 不，特征序列是向下笔，如果趋势向上，它们应该是一底比一底高。如果出现顶分型，说明底不再抬高，反而降低)
                        # 修正：向上线段，特征序列为向下笔。正常延伸时，向下笔的低点应该不断抬高？不对。
                        # 向上线段：笔是 上、下、上、下...
                        # 特征序列（下笔）：下1、下2、下3...
                        # 如果线段延伸，下2应该比下1高（即下2的底 > 下1的底？或者下2的顶 > 下1的顶？）
                        # 标准定义：向上线段，特征序列（下笔）的区间应该是“向上”的。即 T2 > T1.
                        # 如果出现顶分型 (T2 > T1 且 T2 > T3)，说明“向上”趋势终结。
                        # 所以找顶分型是对的。
                        if t2.max >= t1.max and t2.max >= t3.max:
                            is_fx = True
                    else:
                        # 向下线段，找特征序列的底分型
                        if t2.min <= t1.min and t2.min <= t3.min:
                            is_fx = True
                    
                    if is_fx:
                        # 缺口判断 (第一元素与第二元素)
                        has_gap = False
                        if xd_dir == "up":
                            # 向上线段，特征序列顶分型
                            # 缺口：T2 与 T1 之间没有重叠。因为是向上趋势，T2 应该在 T1 之上。
                            # 如果 T2.min > T1.max，则为缺口
                            if t2.min > t1.max: has_gap = True
                        else:
                            # 向下线段，特征序列底分型
                            # 缺口：T2 在 T1 之下。
                            # 如果 T2.max < T1.min，则为缺口
                            if t2.max < t1.min: has_gap = True

                        # 线段结束确认 (参照开发指南.md)
                        is_valid = False
                        
                        # 1. 基础分型有效
                        
                        # 2. 破坏确认 (t3 必须突破 t1 的极值)
                        # 向上线段(找顶分型): T3 必须跌破 T1 的底 (T3.low < T1.low) ?
                        # 开发指南: "return third['low'] < first['low']" (for Up Segment)
                        # 向下线段(找底分型): T3 必须升破 T1 的顶 (T3.high > T1.high)
                        
                        break_condition = False
                        if xd_dir == "up":
                             if t3.min < t1.min: break_condition = True
                        else:
                             if t3.max > t1.max: break_condition = True
                             
                        # 3. 缺口特殊处理
                        # 如果有缺口，必须满足 break_condition (其实标准缠论中，有缺口即为“第二种破坏”，通常需要确认)
                        # 如果无缺口，也建议满足 break_condition 以过滤假突破
                        
                        if has_gap:
                            # 有缺口，必须强力确认 (即 T3 至少要回补缺口，甚至突破 T1)
                            # 原代码逻辑：T3 回补缺口即可 (T3.min <= T1.max for UP)
                            # 开发指南逻辑：似乎更严格
                            # 采用折中方案：必须满足 break_condition
                            if break_condition: is_valid = True
                        else:
                            # 无缺口，标准分型
                            # 是否强制要求 break_condition? 
                            # 严格缠论中，无缺口的分型直接成立。但为了过滤震荡，加上 break_condition 会更稳健。
                            # 原代码没有 break_condition。
                            # 依据开发指南，加上。
                            if break_condition: is_valid = True
                            # 如果不加 break_condition，可能会在震荡中频繁切断。
                            # 但如果严格按照缠论，“顶分型无缺口”即成立。
                            # 考虑到“开发指南”特别提到了 verify_fractals 和 check_break_condition，我们加上它。
                            # 如果不想太严格，可以保留原代码的“无缺口即成立”。
                            # 这里遵循“修正”指令，倾向于更准确/严格的实现。
                            pass

                        if is_valid:
                            # 找到分型顶点对应的笔
                            # t2 包含的笔中，极值笔
                            if xd_dir == "up":
                                peak = max(t2.lines, key=lambda b: b.high)
                                fx_obj = XLFX("ding", t2, [t1, t2, t3], True)
                            else:
                                peak = min(t2.lines, key=lambda b: b.low)
                                fx_obj = XLFX("di", t2, [t1, t2, t3], True)
                            # 记录缺口与形态信息
                            fx_obj.qk = has_gap
                            fx_obj.is_line_bad = False

                            end_idx = peak.index
                            end_bi = self.bis[end_idx]
                            
                            # 创建线段
                            xd = XD(
                                start_bi.start,
                                end_bi.end,
                                start_bi,
                                end_bi,
                                xd_dir,
                                fx_obj if xd_dir == "up" else None,
                                fx_obj if xd_dir == "down" else None,
                                len(self.xds),
                            )
                            if xd_dir == "up":
                                xd.high = end_bi.end.val
                                xd.low = start_bi.start.val
                            else:
                                xd.high = start_bi.start.val
                                xd.low = end_bi.end.val
                            xd.done = True
                            self.xds.append(xd)
                            
                            made = True
                            start_idx = peak.index # 下一段从峰值笔开始
                            break
            
            if bi_pohuai_done:
                continue

            if not made:
                # 生成未完成线段（到当前同向最后一笔）
                if last_same.index > start_bi.index + 1:
                    xd = XD(start_bi.start, last_same.end, start_bi, last_same, xd_dir, None, None, len(self.xds))
                    if xd_dir == "up":
                        xd.high = last_same.end.val
                        xd.low = start_bi.start.val
                    else:
                        xd.high = start_bi.start.val
                        xd.low = last_same.end.val
                    xd.done = False
                    self.xds.append(xd)
                    # 结束所有计算，因为已经到了最后
                    break
                else:
                    # 无法构成线段 (如笔数不足)，尝试下一个笔作为起点
                    start_idx += 1


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
