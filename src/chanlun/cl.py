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
            if new_klines[0].date == last_k.date:
                self.klines[-1] = new_klines[0]
                new_klines = new_klines[1:]
            elif new_klines[0].date < last_k.date:
                idx = -1
                for i, k in enumerate(self.klines):
                    if k.date >= new_klines[0].date:
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

        direction = "up"

        for i in range(1, len(self.klines)):
            k = self.klines[i]
            last_ck = self.cl_klines[-1]

            is_included = False
            if (k.h >= last_ck.h and k.l <= last_ck.l) or (last_ck.h >= k.h and last_ck.l <= k.l):
                is_included = True
            
            if is_included:
                new_h = 0
                new_l = 0
                if direction == "up":
                    new_h = max(last_ck.h, k.h)
                    new_l = max(last_ck.l, k.l)
                else:
                    new_h = min(last_ck.h, k.h)
                    new_l = min(last_ck.l, k.l)
                
                last_ck.h = new_h
                last_ck.l = new_l
                last_ck.k_index = k.index
                last_ck.date = k.date
                last_ck.a += k.a
                last_ck.klines.append(k)
                last_ck.n += 1
            else:
                if k.h > last_ck.h and k.l > last_ck.l:
                    direction = "up"
                elif k.h < last_ck.h and k.l < last_ck.l:
                    direction = "down"
                
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
        
        for i in range(1, len(self.cl_klines) - 1):
            k1 = self.cl_klines[i-1]
            k2 = self.cl_klines[i]
            k3 = self.cl_klines[i+1]
            
            fx_type = None
            if k2.h > k1.h and k2.h > k3.h:
                fx_type = "ding"
            elif k2.l < k1.l and k2.l < k3.l:
                fx_type = "di"
            
            if fx_type:
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
        if start_fx.type == end_fx.type:
            return False
        
        k_diff = abs(end_fx.k.index - start_fx.k.index)
        is_valid = True
        
        if bi_type == Config.BI_TYPE_OLD.value:
            if k_diff < 4: is_valid = False
        elif bi_type == Config.BI_TYPE_NEW.value:
            if k_diff < 3: is_valid = False
        elif bi_type == Config.BI_TYPE_JDB.value:
            if k_diff < 3: is_valid = False
        else:
            if k_diff < 4: is_valid = False
            
        if is_valid:
            if start_fx.type == "ding" and end_fx.val >= start_fx.val: is_valid = False
            if start_fx.type == "di" and end_fx.val <= start_fx.val: is_valid = False
            
            # 检查是否满足 配置的 笔最小K线数量
            if k_diff < self.config.get("bi_min_k_num", 0):
                 is_valid = False
                 
        return is_valid

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
                bi.high = max(start_fx.val, end_fx.val)
                bi.low = min(start_fx.val, end_fx.val)
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
                    last_bi.high = max(last_bi.start.val, next_fx.val)
                    last_bi.low = min(last_bi.start.val, next_fx.val)
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
                bi.high = max(start_fx.val, next_fx.val)
                bi.low = min(start_fx.val, next_fx.val)
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

        # 线段划分 - 特征序列法
        current_xd_start_index = 0
        
        while current_xd_start_index < len(self.bis):
            # 确定当前线段的假设方向
            start_bi = self.bis[current_xd_start_index]
            # 线段的方向与第一笔的方向相同
            xd_direction = start_bi.type
            
            # 特征序列的元素：
            # 向上线段，特征序列为向上的笔 (Standard: "Downward strokes" is for checking termination, 
            # but effectively we look for the peak of Up strokes? 
            # No, standard uses Reverse Direction Strokes.
            # Let's follow the standard: Up Segment -> Feature Sequence = Down Strokes)
            # Wait, if Up Segment, we are looking for a Down Segment to start.
            # A Down Segment starts with a Down Stroke? No, it starts after a Peak.
            # The Peak is established by Up Strokes.
            # Actually, many implementations use "Same Direction Strokes" to find the Peak.
            # Let's use "Same Direction Strokes" (Up Strokes for Up Segment) to find the Fractal.
            # This effectively identifies the Turning Point.
            
            # 收集特征序列（同向笔）
            feature_seq_bis = []
            tzxls = []
            end_index = -1
            found_fractal = False
            
            # 从当前线段开始的笔向后遍历
            # 我们至少需要 3 笔来构成线段 (Start, Opposite, Start...)
            # So index + 1? No.
            # If Up Segment: Up, Down, Up. (3 strokes).
            # The Up strokes are b1, b3.
            # We need to collect Up strokes.
            
            for i in range(current_xd_start_index, len(self.bis)):
                bi = self.bis[i]
                
                if bi.type == xd_direction:
                    # 创建 TZXL
                    pre_line = self.bis[i-2] if i >= 2 else None 
                    # Avoid negative indexing wrapping around
                    if i < 2:
                        pre_line = None
                    
                    # Inclusion Handling Logic
                    # If xd_direction is Up, we use Upward Inclusion (High/High, Low/Low)?
                    # Let's define inclusion direction based on trend.
                    # Up Segment -> Rising -> Upward Inclusion.
                    
                    current_tzxl = TZXL(
                        bh_direction=xd_direction,
                        line=bi,
                        pre_line=None, # We will handle linkage later or ignore
                        line_bad=False,
                        done=True
                    )
                    
                    # Merge with previous TZXL if included
                    if len(tzxls) > 0:
                        last_tzxl = tzxls[-1]
                        
                        # Check Inclusion
                        is_included = False
                        if (current_tzxl.max <= last_tzxl.max and current_tzxl.min >= last_tzxl.min) or \
                           (last_tzxl.max <= current_tzxl.max and last_tzxl.min >= current_tzxl.min):
                            is_included = True
                            
                        if is_included:
                            # Merge
                            # Direction?
                            # If Up Segment (finding Top), we assume Upward Trend.
                            # Upward Inclusion: New High = Max(H1, H2), New Low = Max(L1, L2)
                            # Down Segment (finding Bottom): Downward Inclusion: New High = Min(H1, H2), New Low = Min(L1, L2)
                            
                            if xd_direction == "up":
                                last_tzxl.max = max(last_tzxl.max, current_tzxl.max)
                                last_tzxl.min = max(last_tzxl.min, current_tzxl.min)
                            else:
                                last_tzxl.max = min(last_tzxl.max, current_tzxl.max)
                                last_tzxl.min = min(last_tzxl.min, current_tzxl.min)
                            
                            last_tzxl.lines.append(bi)
                            # Update line to be the one with extreme value?
                            # For Up Segment (Top), we want the Highest High line.
                            if xd_direction == "up":
                                if bi.high > last_tzxl.line.high:
                                    last_tzxl.line = bi
                            else:
                                if bi.low < last_tzxl.line.low:
                                    last_tzxl.line = bi
                        else:
                            tzxls.append(current_tzxl)
                    else:
                        tzxls.append(current_tzxl)
                    
                    # Check for Fractal in TZXLs
                    if len(tzxls) >= 3:
                        t1 = tzxls[-3]
                        t2 = tzxls[-2]
                        t3 = tzxls[-1]
                        
                        is_fractal = False
                        if xd_direction == "up":
                            # Top Fractal
                            if t2.max > t1.max and t2.max > t3.max:
                                is_fractal = True
                        else:
                            # Bottom Fractal
                            if t2.min < t1.min and t2.min < t3.min:
                                is_fractal = True
                                
                        if is_fractal:
                            # Found End of Segment
                            # The segment ends at t2's line end
                            end_bi = t2.line
                            
                            # Valid Segment Check
                            # Must have at least 3 strokes
                            if (end_bi.index - start_bi.index) >= 3:
                                # Also check for "Gap" or "Stroke Destruction" if needed
                                # Simplified: Accept it.
                                
                                end_index = end_bi.index
                                
                                # Construct XLFX for the segment
                                fx_type = "ding" if xd_direction == "up" else "di"
                                xlfx = XLFX(fx_type, t2, [t1, t2, t3], True)
                                
                                xd = XD(
                                    start=start_bi.start,
                                    end=end_bi.end,
                                    start_line=start_bi,
                                    end_line=end_bi,
                                    _type=xd_direction,
                                    index=len(self.xds)
                                )
                                if xd_direction == "up":
                                    xd.ding_fx = xlfx
                                else:
                                    xd.di_fx = xlfx
                                    
                                # Calculate High/Low
                                segment_bis = self.bis[current_xd_start_index : end_index + 1]
                                xd.high = max([b.high for b in segment_bis])
                                xd.low = min([b.low for b in segment_bis])
                                
                                self.xds.append(xd)
                                found_fractal = True
                                
                                # Next segment starts from the stroke AFTER end_bi
                                current_xd_start_index = end_index + 1
                                break
            
            if not found_fractal:
                # No more segments can be found
                break

    def _cal_zsd(self):
        self.zsds = self.xds

    def _cal_zs(self):
        # Clear existing ZSs
        self.bi_zss = {}
        self.xd_zss = {}
        
        # --- Calculate Stroke Pivots (BI ZS) ---
        zs_bi_types = self.config.get("zs_bi_type", [])
        # Ensure it is a list
        if not isinstance(zs_bi_types, list):
            zs_bi_types = [zs_bi_types]
            
        for zs_type in zs_bi_types:
            self.bi_zss[zs_type] = []
            lines = self.bis
            if len(lines) < 3:
                continue
                
            i = 0
            while i <= len(lines) - 3:
                b1 = lines[i]
                b2 = lines[i+1]
                b3 = lines[i+2]
                
                # Pivot Definition: Overlap of 3 consecutive lines
                # ZG = min(Highs)
                # ZD = max(Lows)
                zg = min(b1.high, b2.high, b3.high)
                zd = max(b1.low, b2.low, b3.low)
                
                # Check if valid pivot (ZG > ZD)
                if zg > zd:
                    zs = ZS(
                        zs_type=zs_type,
                        start=b1.start,
                        end=b3.end,
                        zg=zg,
                        zd=zd,
                        gg=max(b1.high, b2.high, b3.high),
                        dd=min(b1.low, b2.low, b3.low),
                        index=len(self.bi_zss[zs_type]),
                        line_num=3,
                        _type="up" if b1.type == "down" else "down"
                    )
                    zs.lines = [b1, b2, b3]
                    zs.real = True
                    self.bi_zss[zs_type].append(zs)
                    
                    # Extension: Add subsequent lines if they overlap with [ZD, ZG]
                    j = i + 3
                    while j < len(lines):
                        bn = lines[j]
                        # Check overlap: Not (High < ZD or Low > ZG)
                        if not (bn.high < zd or bn.low > zg):
                            zs.lines.append(bn)
                            zs.end = bn.end
                            # Update GG/DD (Highest/Lowest of the whole pivot oscillation)
                            if bn.high > zs.gg: zs.gg = bn.high
                            if bn.low < zs.dd: zs.dd = bn.low
                            j += 1
                        else:
                            break
                    # Continue from the end of this pivot? 
                    # Standard: The next pivot starts after this one.
                    # The last line of the pivot (bn-1) is part of the pivot.
                    # The next 3 lines start from...
                    # Usually we skip the lines consumed by the pivot.
                    # But we can share lines? "Expansion"?
                    # Simplified: Skip to j-1 (the last line of pivot)
                    # i = j - 1 ? No, next pivot needs 3 new lines or share?
                    # Let's set i = j (start from the line that failed to include)
                    # Wait, if pivot is lines [0, 1, 2, 3, 4]. Line 5 failed.
                    # Next check starts at 5? Yes.
                    i = j - 1 # Backtrack one? No. 
                    # Line 5 (index j) is the one that BROKE the pivot condition.
                    # It is the "Leaving Stroke".
                    # Can Line 5 be the start of a new Pivot? Yes.
                    # So i = j.
                    # Wait, Line 5 might be the 'connector'.
                    # Let's set i = j - 1 to be safe? 
                    # If line 5 is the first of next 3.
                    # Previous pivot used lines up to j-1.
                    # Next pivot starts at j-1? No, j-1 is inside pivot.
                    # Next pivot starts at j?
                    # Let's assume standard non-sharing (except connection point).
                    # Connection point: The end of Pivot 1 is Start of Pivot 2?
                    # Pivot 1 end FX is `zs.end`.
                    # The line causing break is `lines[j]`.
                    # So we start checking from `lines[j]`.
                    # But wait, `lines[j]` is just 1 line. We need 3.
                    # So we reset i to j.
                    i = j - 1 # Re-evaluate from the last line of the pivot?
                    # Actually, in strict non-overlap of pivots (except sharing the connecting stroke):
                    # Pivot ends at line[j-1].
                    # Next pivot can start at line[j-1]?
                    # No, standard: Pivot 1 + Connector + Pivot 2.
                    # So i = j.
                    # However, let's use i = j-1 to allow "Continuous" pivots check (e.g. if line[j-1] can be start of next).
                    # Actually, strict definition: ZS is A+B+C.
                    # If A,B,C,D,E form ZS.
                    # F breaks.
                    # Next check F, G, H.
                    i = j - 1
                else:
                    i += 1

        # --- Calculate Segment Pivots (XD ZS) ---
        zs_xd_types = self.config.get("zs_xd_type", [])
        if not isinstance(zs_xd_types, list):
            zs_xd_types = [zs_xd_types]

        for zs_xd_type in zs_xd_types:
            self.xd_zss[zs_xd_type] = []
            lines = self.xds
            if len(lines) < 3:
                continue

            i = 0
            while i <= len(lines) - 3:
                x1 = lines[i]
                x2 = lines[i+1]
                x3 = lines[i+2]

                zg = min(x1.high, x2.high, x3.high)
                zd = max(x1.low, x2.low, x3.low)

                if zg > zd:
                    zs = ZS(
                        zs_type=zs_xd_type,
                        start=x1.start,
                        end=x3.end,
                        zg=zg,
                        zd=zd,
                        gg=max(x1.high, x2.high, x3.high),
                        dd=min(x1.low, x2.low, x3.low),
                        index=len(self.xd_zss[zs_xd_type]),
                        line_num=3,
                        _type="up" if x1.type == "down" else "down"
                    )
                    zs.lines = [x1, x2, x3]
                    zs.real = True
                    self.xd_zss[zs_xd_type].append(zs)

                    j = i + 3
                    while j < len(lines):
                        xn = lines[j]
                        if not (xn.high < zd or xn.low > zg):
                            zs.lines.append(xn)
                            zs.end = xn.end
                            if xn.high > zs.gg: zs.gg = xn.high
                            if xn.low < zs.dd: zs.dd = xn.low
                            j += 1
                        else:
                            break
                    i = j - 1
                else:
                    i += 1

    def _cal_mmd_bc(self):
        # Calculate MACD LD (Force) and Check BC (Divergence)
        # And Identify MMD (Buy/Sell Points)
        
        # --- 1. Calculate for Strokes (BI) ---
        zs_type = self.config.get("zs_bi_type", ["common"])[0]
        zss = self.bi_zss.get(zs_type, [])
        
        for i in range(len(self.bis)):
            bi = self.bis[i]
            
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
            if not zss:
                continue
                
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
                    # Check if prev_bi is the leaving stroke
                    # The leaving stroke starts exactly at the pivot's end fractal.
                    if prev_bi.start.index == zs.end.index:
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
                    if bi.bc_exists(["qs", "pz"], zs_type): 
                        if bi.type == "down" and bi.low < zs.dd:
                            bi.add_mmd("1buy", zs, zs_type)
                        if bi.type == "up" and bi.high > zs.gg:
                            bi.add_mmd("1sell", zs, zs_type)
                    elif has_prev_3sell or has_prev_3buy:
                         # If following a 3rd Buy/Sell, we can be slightly more lenient but still prefer divergence
                         # Check if current stroke is divergence (bi)
                         if bi.bc_exists(["bi"], zs_type):
                             if bi.type == "down" and (bi.low < zs.dd or has_prev_3sell):
                                bi.add_mmd("1buy", zs, zs_type)
                             if bi.type == "up" and (bi.high > zs.gg or has_prev_3buy):
                                bi.add_mmd("1sell", zs, zs_type)

                # 2nd Buy/Sell
                # 1. Two centers trend, new low, then next segment no new low 2B
                # 2. Two centers trend, no new low, segment divergence 2B
                # 3. After 1B, no new low 2B
                # 4. After 3S no 1B produced, subsequent no new low also appears 2B
                if i >= 2:
                    prev_bi_2 = self.bis[i-2]
                    # Check if prev_bi_2 was a 1st Buy
                    is_1buy = any(m.name == "1buy" for m in prev_bi_2.get_mmds(zs_type))
                    is_1sell = any(m.name == "1sell" for m in prev_bi_2.get_mmds(zs_type))

                    if bi.type == "down":
                        # Case 1 & 3: After 1buy
                        if is_1buy and bi.low > prev_bi_2.low:
                            bi.add_mmd("2buy", prev_bi_2.get_mmds(zs_type)[0].zs, zs_type)
                        
                        # Case 4: After 3S, no 1B, no new low
                        elif i >= 3:
                            prev_bi_3 = self.bis[i-3] # The UP stroke that might be 3S
                            has_3sell_prev = any(m.name == "3sell" for m in prev_bi_3.get_mmds(zs_type))
                            if has_3sell_prev and not is_1buy and bi.low > prev_bi_2.low:
                                bi.add_mmd("2buy", zs, zs_type)

                    if bi.type == "up":
                        # Case 1 & 3
                        if is_1sell and bi.high < prev_bi_2.high:
                            bi.add_mmd("2sell", prev_bi_2.get_mmds(zs_type)[0].zs, zs_type)
                        
                        # Case 4
                        elif i >= 3:
                            prev_bi_3 = self.bis[i-3]
                            has_3buy_prev = any(m.name == "3buy" for m in prev_bi_3.get_mmds(zs_type))
                            if has_3buy_prev and not is_1sell and bi.high < prev_bi_2.high:
                                bi.add_mmd("2sell", zs, zs_type)

                # Class 2 Buy/Sell (类二买/卖)
                # Logic: Previous same-dir stroke had 2buy. Current stroke overlaps and Higher Low.
                if i >= 2:
                    prev_bi_2 = self.bis[i-2]
                    prev_bi = self.bis[i-1]
                    if bi.type == "down":
                        # Check if prev_bi_2 has 2buy
                        has_2buy = any(m.name == "2buy" for m in prev_bi_2.get_mmds(zs_type))
                        if has_2buy:
                            # Check overlap (Pivot formation) -> ZG > ZD
                            # ZG = min(prev_bi_2.high, prev_bi.high, bi.high)
                            # ZD = max(prev_bi_2.low, prev_bi.low, bi.low) -> Since prev_bi.low == prev_bi_2.low, and we need bi.low > prev_bi_2.low
                            # So ZD = bi.low
                            # We need min(Highs) > bi.low
                            zg = min(prev_bi_2.high, prev_bi.high, bi.high)
                            if zg > bi.low and bi.low > prev_bi_2.low:
                                 bi.add_mmd("l2buy", zs, zs_type)
                    if bi.type == "up":
                        has_2sell = any(m.name == "2sell" for m in prev_bi_2.get_mmds(zs_type))
                        if has_2sell:
                            # ZG = min(Lows) ? No, for Down Pivot (Sell), we check overlap.
                            # ZG = min(Highs) = bi.high (since bi.high < prev_bi_2.high)
                            # ZD = max(Lows)
                            # We need ZG > ZD? No, for Sell Pivot (Upward overlap):
                            # Ranges: [L1, H1], [L2, H2], [L3, H3].
                            # Overlap exists if min(H) > max(L).
                            # Here we are looking for Lower High.
                            # zd = max(prev_bi_2.low, prev_bi.low, bi.low)
                            # if bi.high > zd and bi.high < prev_bi_2.high:
                            # Actually simplified:
                            zd = max(prev_bi_2.low, prev_bi.low, bi.low)
                            if bi.high > zd and bi.high < prev_bi_2.high:
                                 bi.add_mmd("l2sell", zs, zs_type)

                # Class 3 Buy/Sell (类三买/卖)
                # Logic: Previous same-dir stroke had 3buy. Current stroke overlaps and Higher Low.
                if i >= 2:
                    prev_bi_2 = self.bis[i-2]
                    prev_bi = self.bis[i-1]
                    if bi.type == "down":
                        has_3buy = any(m.name == "3buy" for m in prev_bi_2.get_mmds(zs_type))
                        if has_3buy:
                            zg = min(prev_bi_2.high, prev_bi.high, bi.high)
                            if zg > bi.low and bi.low > prev_bi_2.low:
                                 bi.add_mmd("l3buy", zs, zs_type)
                    if bi.type == "up":
                        has_3sell = any(m.name == "3sell" for m in prev_bi_2.get_mmds(zs_type))
                        if has_3sell:
                            zd = max(prev_bi_2.low, prev_bi.low, bi.low)
                            if bi.high > zd and bi.high < prev_bi_2.high:
                                 bi.add_mmd("l3sell", zs, zs_type)
        
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

                    if xd.bc_exists(["qs", "pz"], zs_xd_type):
                        if xd.type == "down" and xd.low < zs.dd: xd.add_mmd("1buy", zs, zs_xd_type)
                        if xd.type == "up" and xd.high > zs.gg: xd.add_mmd("1sell", zs, zs_xd_type)
                    elif xd.bc_exists(["xd"], zs_xd_type):
                        if xd.type == "down":
                             if xd.low < zs.dd or has_prev_3sell: xd.add_mmd("1buy", zs, zs_xd_type)
                        if xd.type == "up":
                             if xd.high > zs.gg or has_prev_3buy: xd.add_mmd("1sell", zs, zs_xd_type)
                        
                    # 2nd Buy/Sell
                    if i >= 2:
                        prev_xd_2 = self.xds[i-2]
                        if xd.type == "down":
                            # Case 1 & 3
                            one_buys = [m for m in prev_xd_2.get_mmds(zs_xd_type) if m.name == "1buy"]
                            if one_buys and xd.low > prev_xd_2.low: xd.add_mmd("2buy", one_buys[0].zs, zs_xd_type)
                            
                            # Case 2: Truncation
                            if xd.low > zs.dd and (xd.bc_exists(['qs', 'pz', 'xd'], zs_xd_type)):
                                 xd.add_mmd("2buy", zs, zs_xd_type)

                            # Case 4: After 3S
                            if i >= 3:
                                prev_xd_3 = self.xds[i-3]
                                has_3sell_prev = any(m.name == "3sell" for m in prev_xd_3.get_mmds(zs_xd_type))
                                has_1buy_prev = any(m.name == "1buy" for m in prev_xd_2.get_mmds(zs_xd_type))
                                if has_3sell_prev and not has_1buy_prev and xd.low > prev_xd_2.low:
                                    xd.add_mmd("2buy", zs, zs_xd_type)

                        if xd.type == "up":
                            # Case 1 & 3
                            one_sells = [m for m in prev_xd_2.get_mmds(zs_xd_type) if m.name == "1sell"]
                            if one_sells and xd.high < prev_xd_2.high: xd.add_mmd("2sell", one_sells[0].zs, zs_xd_type)

                            # Case 2: Truncation
                            if xd.high < zs.gg and (xd.bc_exists(['qs', 'pz', 'xd'], zs_xd_type)):
                                 xd.add_mmd("2sell", zs, zs_xd_type)
                            
                            # Case 4: After 3B
                            if i >= 3:
                                prev_xd_3 = self.xds[i-3]
                                has_3buy_prev = any(m.name == "3buy" for m in prev_xd_3.get_mmds(zs_xd_type))
                                has_1sell_prev = any(m.name == "1sell" for m in prev_xd_2.get_mmds(zs_xd_type))
                                if has_3buy_prev and not has_1sell_prev and xd.high < prev_xd_2.high:
                                    xd.add_mmd("2sell", zs, zs_xd_type)
                            
                    # Class 2/3
                    if i >= 2:
                        prev_xd_2 = self.xds[i-2]
                        prev_xd = self.xds[i-1]
                        if xd.type == "down":
                            if any(m.name == "2buy" for m in prev_xd_2.get_mmds(zs_xd_type)):
                                zg = min(prev_xd_2.high, prev_xd.high, xd.high)
                                if zg > xd.low and xd.low > prev_xd_2.low: xd.add_mmd("l2buy", zs, zs_xd_type)
                            if any(m.name == "3buy" for m in prev_xd_2.get_mmds(zs_xd_type)):
                                zg = min(prev_xd_2.high, prev_xd.high, xd.high)
                                if zg > xd.low and xd.low > prev_xd_2.low: xd.add_mmd("l3buy", zs, zs_xd_type)
                        if xd.type == "up":
                            if any(m.name == "2sell" for m in prev_xd_2.get_mmds(zs_xd_type)):
                                zd = max(prev_xd_2.low, prev_xd.low, xd.low)
                                if xd.high > zd and xd.high < prev_xd_2.high: xd.add_mmd("l2sell", zs, zs_xd_type)
                            if any(m.name == "3sell" for m in prev_xd_2.get_mmds(zs_xd_type)):
                                zd = max(prev_xd_2.low, prev_xd.low, xd.low)
                                if xd.high > zd and xd.high < prev_xd_2.high: xd.add_mmd("l3sell", zs, zs_xd_type)

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
                i = j
            else:
                i += 1
        return zss

    def beichi_pz(self, zs: ZS, now_line: LINE) -> Tuple[bool, Union[LINE, None]]:
        if len(zs.lines) < 1:
            return False, None
        # 简单的盘整背驰判断：离开段力度 < 进入段力度
        # 由于无法直接获取进入段，这里尝试获取中枢第一段的反向段作为参考（假设是进入段）
        # 或者比较 离开段 与 中枢内同向段 的力度
        
        # 获取中枢内同向段
        same_dir_lines = [l for l in zs.lines if l.type == now_line.type]
        if not same_dir_lines:
            return False, None
            
        # 比较最后一段同向段
        last_same = same_dir_lines[-1]
        
        ld1 = last_same.get_ld(self)
        ld2 = now_line.get_ld(self)
        
        if compare_ld_beichi(ld1, ld2, now_line.type):
            return True, last_same
            
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
             
        # 寻找连接段 (Impulse 1)
        # 连接段在 zs1.end 到 zs2.start 之间
        # 简单处理：找到 lines 中位于 zs1 和 zs2 之间的最大力度的线
        connect_lines = [l for l in lines if l.index > zs1.lines[-1].index and l.index < zs2.lines[0].index and l.type == now_line.type]
        
        if not connect_lines:
            return False, []
            
        # 取力度最大的连接段（通常连接段就是一笔，但如果有复杂结构可能多笔）
        # 这里假设连接段就是 zs2 之前的同向段
        c_line = connect_lines[-1]
        
        ld1 = c_line.get_ld(self)
        ld2 = now_line.get_ld(self)
        
        if compare_ld_beichi(ld1, ld2, now_line.type):
            return True, [c_line]
            
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
