var TvIdxTLZMACD = (function () {
  return {
    idx: function (PineJS) {
      return {
        name: "趋势浪子MACD",
        metainfo: {
          _metainfoVersion: 53,
          id: "CustomIndicatorsTLZMACD@tv-basicstudies-1",
          description: "趋势浪子体系专用MACD（10/20/5）+ 金死叉 + 顶底背离",
          shortDescription: "趋势浪子MACD (10/20/5) 背离版",
          is_price_study: false,
          isCustomIndicator: true,
          plots: [
            { id: "plot_hist_up", type: "line" },
            { id: "plot_hist_dn", type: "line" },
            { id: "plot_dif", type: "line" },
            { id: "plot_dea", type: "line" },
            { id: "plot_crossGold", type: "shapes" },
            { id: "plot_crossDead", type: "shapes" },
            { id: "plot_bullShape", type: "shapes" },
            { id: "plot_bearShape", type: "shapes" },
          ],
          defaults: {
            palettes: {},
            styles: {
              plot_hist_up: {
                linestyle: 0, linewidth: 1, plottype: 1,
                trackPrice: false, transparency: 0, visible: true,
                color: "#ef232a",
              },
              plot_hist_dn: {
                linestyle: 0, linewidth: 1, plottype: 1,
                trackPrice: false, transparency: 0, visible: true,
                color: "#14b143",
              },
              plot_dif: {
                linestyle: 0, linewidth: 1, plottype: 0,
                trackPrice: false, transparency: 0, visible: true,
                color: "#FFFFFF",
              },
              plot_dea: {
                linestyle: 0, linewidth: 1, plottype: 0,
                trackPrice: false, transparency: 0, visible: true,
                color: "#FFD700",
              },
              plot_crossGold: {
                color: "#FF5252", textColor: "#FF5252",
                plottype: "shape_xcross", location: "Absolute", visible: true,
              },
              plot_crossDead: {
                color: "#00FF00", textColor: "#00FF00",
                plottype: "shape_xcross", location: "Absolute", visible: true,
              },
              plot_bullShape: {
                color: "#4CAF50", textColor: "#4CAF50",
                plottype: "shape_triangle_up", location: "Absolute", visible: true,
              },
              plot_bearShape: {
                color: "#F44336", textColor: "#F44336",
                plottype: "shape_triangle_down", location: "Absolute", visible: true,
              },
            },
            inputs: {
              fast_length: 10,
              slow_length: 20,
              signal_length: 5,
              price_thresh: 0.0005,
              require_same_side: true,
              require_dif_zero: true,
              plotGold: true,
              plotDead: true,
              plotBull: true,
              plotBear: true,
            },
          },
          palettes: {},
          styles: {
            plot_hist_up: { title: "能量柱(水上)", histogramBase: 0 },
            plot_hist_dn: { title: "能量柱(水下)", histogramBase: 0 },
            plot_dif: { title: "DIF快线", histogramBase: 0 },
            plot_dea: { title: "DEA慢线", histogramBase: 0 },
            plot_crossGold: { title: "金叉", location: "Absolute" },
            plot_crossDead: { title: "死叉", location: "Absolute" },
            plot_bullShape: { title: "底背离", location: "Absolute", text: "底背离" },
            plot_bearShape: { title: "顶背离", location: "Absolute", text: "顶背离" },
          },
          inputs: [
            {
              id: "fast_length",
              name: "快线(short=10,紫色级别)",
              type: "integer",
              defval: 10,
              min: 1,
              max: 200,
            },
            {
              id: "slow_length",
              name: "慢线(long=20,白色级别零轴)",
              type: "integer",
              defval: 20,
              min: 1,
              max: 400,
            },
            {
              id: "signal_length",
              name: "M值(信号线=5)",
              type: "integer",
              defval: 5,
              min: 1,
              max: 100,
            },
            {
              id: "price_thresh",
              name: "价格创新幅度阈值(%)",
              type: "float",
              defval: 0.0005,
              min: 0.0,
              max: 0.1,
              step: 0.0005,
            },
            {
              id: "require_same_side",
              name: "要求MACD柱子同侧(过滤零轴附近背离)",
              type: "bool",
              defval: true,
            },
            {
              id: "require_dif_zero",
              name: "要求DIF远离零轴(顶背离DIF>0,底背离DIF<0)",
              type: "bool",
              defval: true,
            },
            { id: "plotGold", name: "显示金叉", type: "bool", defval: true },
            { id: "plotDead", name: "显示死叉", type: "bool", defval: true },
            { id: "plotBull", name: "显示底背离", type: "bool", defval: true },
            { id: "plotBear", name: "显示顶背离", type: "bool", defval: true },
          ],
          format: {
            type: "price",
            precision: 4,
          },
        },
        constructor: function () {
          // ================================================================
          // MACD 背离算法：比较中枢前后同向段的 DIF 极值与柱子总面积。
          // 仅单根反向柱视为小级别缠绕，并入原同向段。
          // 顶背离：价格创新高，DIF 高点降低或红柱总面积缩小。
          // 底背离：价格创新低或近似同低，DIF 低点抬高或绿柱总面积缩小。
          // ================================================================

          this._frameCount = 0;
          this._positiveSegments = [];
          this._negativeSegments = [];
          this._activeSegment = null;
          this._pendingSegment = null;

          this.init = function (context, inputCallback) {
            this._frameCount = 0;
            this._positiveSegments = [];
            this._negativeSegments = [];
            this._activeSegment = null;
            this._pendingSegment = null;
          };

          this.main = function (context, inputCallback) {
            this._context = context;
            this._input = inputCallback;
            this._frameCount++;

            var fast_length = this._input(0);
            var slow_length = this._input(1);
            var signal_length = this._input(2);
            var price_thresh = this._input(3);
            var require_same_side = this._input(4);
            var require_dif_zero = this._input(5);
            var plotGold = this._input(6);
            var plotDead = this._input(7);
            var plotBull = this._input(8);
            var plotBear = this._input(9);

            var h = this._context.new_var(PineJS.Std.high(this._context));
            var l = this._context.new_var(PineJS.Std.low(this._context));
            var c = this._context.new_var(PineJS.Std.close(this._context));

            // 趋势浪子MACD：DIF = EMA(fast) - EMA(slow)
            var fast_ema = PineJS.Std.ema(c, fast_length, this._context);
            var slow_ema = PineJS.Std.ema(c, slow_length, this._context);
            var dif = this._context.new_var(fast_ema - slow_ema);
            var dea = this._context.new_var(
              PineJS.Std.ema(dif, signal_length, this._context)
            );

            var histVal = (dif.get(0) - dea.get(0)) * 2;

            // 能量柱水上水下着色
            var histUp = histVal >= 0 ? histVal : NaN;
            var histDn = histVal < 0 ? histVal : NaN;

            // ========== 金叉死叉检测 ==========
            var difCurr = dif.get(0);
            var difPrev = dif.get(1);
            var deaCurr = dea.get(0);
            var deaPrev = dea.get(1);
            var isGoldCross = difPrev <= deaPrev && difCurr > deaCurr;
            var isDeadCross = difPrev >= deaPrev && difCurr < deaCurr;
            // 用前后两根的线性插值计算实际交点纵坐标，避免标记偏在线的一侧。
            var prevGap = difPrev - deaPrev;
            var currGap = difCurr - deaCurr;
            var crossRatio = Math.abs(prevGap) / (Math.abs(prevGap) + Math.abs(currGap));
            var crossValue = difPrev + (difCurr - difPrev) * crossRatio;
            var crossGoldVal = (plotGold && isGoldCross) ? crossValue : NaN;
            var crossDeadVal = (plotDead && isDeadCross) ? crossValue : NaN;

            // ========== 顶底背离检测（中枢前后同向段力度比较） ==========
            var bullShape = NaN;
            var bearShape = NaN;

            // 逐 K 线维护柱段，避免动态深度回看取不到历史数据。
            function newSegment(sign, high, low, difValue, histValue) {
              return {
                sign: sign,
                price: sign > 0 ? high : low,
                difExtreme: difValue,
                area: Math.abs(histValue),
                bars: 1,
              };
            }
            function updateSegment(segment, high, low, difValue, histValue) {
              var priceValue = segment.sign > 0 ? high : low;
              if (segment.sign > 0) {
                if (priceValue > segment.price) segment.price = priceValue;
                if (difValue > segment.difExtreme) segment.difExtreme = difValue;
              } else {
                if (priceValue < segment.price) segment.price = priceValue;
                if (difValue < segment.difExtreme) segment.difExtreme = difValue;
              }
              segment.area += Math.abs(histValue);
              segment.bars++;
            }
            function isPriceHighOrNear(curr, prev) {
              return curr >= prev - Math.max(Math.abs(prev), 1) * price_thresh;
            }
            function isPriceLowOrNear(curr, prev) {
              return curr <= prev + Math.max(Math.abs(prev), 1) * price_thresh;
            }

            var histSign = histVal >= 0 ? 1 : -1;
            var completedSegment = null;
            var confirmedCrossValue = NaN;
            if (this._activeSegment === null) {
              this._activeSegment = newSegment(histSign, h.get(0), l.get(0), difCurr, histVal);
            } else if (histSign === this._activeSegment.sign) {
              // 单根反向柱是假交叉：价格纳入原段，反向柱面积不计入同向力度。
              if (this._pendingSegment !== null) {
                var pendingPrice = this._activeSegment.sign > 0
                  ? this._pendingSegment.high : this._pendingSegment.low;
                if (this._activeSegment.sign > 0 && pendingPrice > this._activeSegment.price) {
                  this._activeSegment.price = pendingPrice;
                }
                if (this._activeSegment.sign < 0 && pendingPrice < this._activeSegment.price) {
                  this._activeSegment.price = pendingPrice;
                }
                this._pendingSegment = null;
              }
              updateSegment(this._activeSegment, h.get(0), l.get(0), difCurr, histVal);
            } else {
              if (this._pendingSegment === null) {
                this._pendingSegment = {
                  segment: newSegment(histSign, h.get(0), l.get(0), difCurr, histVal),
                  high: h.get(0),
                  low: l.get(0),
                  crossValue: crossValue,
                };
              } else {
                updateSegment(this._pendingSegment.segment, h.get(0), l.get(0), difCurr, histVal);
                this._pendingSegment.high = Math.max(this._pendingSegment.high, h.get(0));
                this._pendingSegment.low = Math.min(this._pendingSegment.low, l.get(0));
                // 连续两根反向柱确认换段；标记回绘到第一根反向柱。
                completedSegment = this._activeSegment;
                confirmedCrossValue = this._pendingSegment.crossValue;
                this._activeSegment = this._pendingSegment.segment;
                this._pendingSegment = null;
              }
            }

            if (completedSegment !== null) {
              var segmentList = completedSegment.sign > 0
                ? this._positiveSegments : this._negativeSegments;
              var previousSegment = segmentList.length > 0
                ? segmentList[segmentList.length - 1] : null;

              if (previousSegment !== null) {
                if (completedSegment.sign > 0) {
                  var topDifOk = !require_dif_zero ||
                    (completedSegment.difExtreme > 0 && previousSegment.difExtreme > 0);
                  var topMomentumWeak = completedSegment.difExtreme < previousSegment.difExtreme ||
                    completedSegment.area < previousSegment.area;
                  if (plotBear && isPriceHighOrNear(completedSegment.price, previousSegment.price) &&
                      topMomentumWeak && topDifOk) {
                    bearShape = { value: confirmedCrossValue, offset: -1 };
                  }
                } else {
                  var bottomDifOk = !require_dif_zero ||
                    (completedSegment.difExtreme < 0 && previousSegment.difExtreme < 0);
                  var bottomMomentumWeak = completedSegment.difExtreme > previousSegment.difExtreme ||
                    completedSegment.area < previousSegment.area;
                  if (plotBull && isPriceLowOrNear(completedSegment.price, previousSegment.price) &&
                      bottomMomentumWeak && bottomDifOk) {
                    bullShape = { value: confirmedCrossValue, offset: -1 };
                  }
                }
              }

              segmentList.push(completedSegment);
              if (segmentList.length > 10) segmentList.shift();
            }

            return [
              histUp,         // 0: 水上红柱
              histDn,         // 1: 水下绿柱
              difCurr,        // 2: DIF快线
              deaCurr,        // 3: DEA慢线
              crossGoldVal,   // 4: 金叉标注
              crossDeadVal,   // 5: 死叉标注
              bullShape,      // 6: 底背离标注
              bearShape,      // 7: 顶背离标注
            ];
          };
        },
      };
    },
  };
})();
