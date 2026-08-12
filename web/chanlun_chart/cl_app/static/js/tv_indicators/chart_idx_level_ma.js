var TvIdxLevelMA = (function () {
  return {
    idx: function (PineJS) {
      return {
        name: "级别均线(双倍均线体系)",
        metainfo: {
          _metainfoVersion: 53,
          id: "CustomIndicatorsLevelMA@tv-basicstudies-1",
          description: "趋势浪子双倍均线体系（紫10/白20/蓝40/绿80/红160/黄320）",
          shortDescription: "级别均线 MA10/20/40/80/160/320",
          is_price_study: true,
          isCustomIndicator: true,
          plots: [
            { id: "plot_ma10", type: "line" },
            { id: "plot_ma20", type: "line" },
            { id: "plot_ma40", type: "line" },
            { id: "plot_ma80", type: "line" },
            { id: "plot_ma160", type: "line" },
            { id: "plot_ma320", type: "line" },
          ],
          defaults: {
            palettes: {},
            styles: {
              plot_ma10: {
                linestyle: 0, linewidth: 1, plottype: 0,
                trackPrice: false, transparency: 10, visible: true,
                color: "#9932CC", // 紫色 - 内部次级别
              },
              plot_ma20: {
                linestyle: 0, linewidth: 1, plottype: 0,
                trackPrice: false, transparency: 10, visible: true,
                color: "#CCCCCC", // 白色(浅灰) - 本级别一笔
              },
              plot_ma40: {
                linestyle: 0, linewidth: 1, plottype: 0,
                trackPrice: false, transparency: 10, visible: true,
                color: "#4169E1", // 皇家蓝 - 中级别一笔(MACD零轴)
              },
              plot_ma80: {
                linestyle: 0, linewidth: 1, plottype: 0,
                trackPrice: false, transparency: 10, visible: true,
                color: "#008000", // 绿色 - 大级别一笔
              },
              plot_ma160: {
                linestyle: 0, linewidth: 2, plottype: 0,
                trackPrice: false, transparency: 10, visible: true,
                color: "#FF0000", // 红色 - 线段级别(操作级别)
              },
              plot_ma320: {
                linestyle: 0, linewidth: 2, plottype: 0,
                trackPrice: false, transparency: 10, visible: true,
                color: "#FFD700", // 黄色 - 趋势级别(牛熊分界)
              },
            },
            inputs: {},
          },
          palettes: {},
          styles: {
            plot_ma10: { title: "MA10 紫色(内部次级别)" },
            plot_ma20: { title: "MA20 白色(本级别一笔)" },
            plot_ma40: { title: "MA40 蓝色(中级别/零轴)" },
            plot_ma80: { title: "MA80 绿色(大级别一笔)" },
            plot_ma160: { title: "MA160 红色(线段级别)" },
            plot_ma320: { title: "MA320 黄色(趋势级别)" },
          },
          inputs: [],
          format: {
            type: "price",
            precision: 2,
          },
        },
        constructor: function () {
          this.init = function (context, inputCallback) {
            this._context = context;
            this._input = inputCallback;
          };
          this.main = function (context, inputCallback) {
            this._context = context;
            this._input = inputCallback;

            const c = this._context.new_var(PineJS.Std.close(this._context));

            const ma10 = PineJS.Std.sma(c, 10, this._context);
            const ma20 = PineJS.Std.sma(c, 20, this._context);
            const ma40 = PineJS.Std.sma(c, 40, this._context);
            const ma80 = PineJS.Std.sma(c, 80, this._context);
            const ma160 = PineJS.Std.sma(c, 160, this._context);
            const ma320 = PineJS.Std.sma(c, 320, this._context);

            return [
              ma10,   // 0: MA10 紫色 - 内部次级别
              ma20,   // 1: MA20 白色 - 本级别一笔
              ma40,   // 2: MA40 蓝色 - 中级别(MACD零轴)
              ma80,   // 3: MA80 绿色 - 大级别一笔
              ma160,  // 4: MA160 红色 - 线段级别
              ma320,  // 5: MA320 黄色 - 趋势级别
            ];
          };
        },
      };
    },
  };
})();
