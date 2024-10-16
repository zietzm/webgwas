import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import HighchartsBoost from "highcharts/modules/boost";
import { PvaluesResult } from "../lib/api";
if (typeof Highcharts === "object") {
  HighchartsBoost(Highcharts);
  Highcharts.AST.allowedTags.push("input");
  Highcharts.AST.allowedAttributes.push("rel");
  Highcharts.AST.allowedAttributes.push("onmousedown");
}
import React from "react";

export function Docs() {
  return (
    <p className="text-gray-800 dark:text-gray-400">
      Only variants with p &lt; 0.05 are shown. Unfiltered results are available
      for download above.
    </p>
  );
}

interface ColorMapItem {
  phenotype: string;
  color: string;
}

interface LegendProps {
  colorMap: Array<ColorMapItem>;
  title?: string;
}

const Legend: React.FC<LegendProps> = ({ colorMap, title = "Shared hits" }) => {
  return (
    <div className="flex flex-row gap-4">
      {title && <h3 className="text-lg font-semibold">{title}</h3>}
      <div className="flex flex-wrap gap-4">
        {colorMap.map(({ phenotype, color }) => (
          <div key={phenotype} className="flex items-center">
            <div
              className="w-4 h-4 mr-2"
              style={{ backgroundColor: color }}
            ></div>
            <span>{phenotype}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default function ManhattanPlot({ data }: { data: PvaluesResult }) {
  const mainColors = ["rgba(89, 130, 179, .25)", "rgba(64, 98, 140, .25)"];
  const highlightColors = [
    "#ea5545",
    "#f46a9b",
    "#ef9b20",
    "#edbf33",
    "#ede15b",
    "#bdcf32",
    "#87bc45",
    "#27aeef",
    "#b33dc6",
  ];

  const chrPositions = data.chromosome_positions.map((chr) => chr.midpoint);

  const chrIdxToLabel: Map<number, string> = new Map();
  data.chromosome_positions.forEach((chr) => {
    chrIdxToLabel.set(chr.midpoint, chr.chromosome);
  });

  const colorIdxToColor: Map<number, string> = new Map();
  data.color_map.forEach((_, colorIdx) => {
    if (colorIdx < 23) {
      colorIdxToColor.set(colorIdx, mainColors[colorIdx % mainColors.length]);
    } else {
      colorIdxToColor.set(
        colorIdx,
        highlightColors[(colorIdx - 23) % highlightColors.length],
      );
    }
  });

  const variants = data.pvalues.map((p) => {
    return {
      x: p.index,
      y: p.pvalue,
      color: colorIdxToColor.get(p.color)!,
      label: p.label,
    };
  });

  const colorMap: Array<ColorMapItem> = [];
  data.color_map.forEach((phenotype, colorIdx) => {
    if (colorIdx >= 23) {
      colorMap.push({ phenotype, color: colorIdxToColor.get(colorIdx)! });
    }
  });

  const options = {
    credits: {
      enabled: false,
    },
    chart: {
      type: "scatter",
      zoomType: "xy",
    },
    boost: {
      useGPUTranslations: true,
      usePreAllocated: true,
    },
    title: {
      text: null,
    },
    xAxis: {
      gridLineWidth: 1,
      title: {
        text: "Variant position",
      },
      tickPositions: chrPositions,
      labels: {
        overflow: false,
        formatter: function (this: any, _ctx: any): any {
          return chrIdxToLabel.get(this.value);
        },
      },
      showLastLabel: true,
    },
    yAxis: {
      minPadding: 0,
      maxPadding: 0,
      title: {
        useHTML: true,
        text: "Negative log<sub>10</sub> p-value",
      },
      plotLines: [
        {
          color: "#FF0000",
          width: 2,
          value: 7.3,
          dashStyle: "dash",
        },
      ],
    },
    legend: {
      enabled: false,
    },
    tooltip: {
      followPointer: false,
      headerFormat: "",
      pointFormat: `
        <div>
          <p style="user-select: text; cursor: text;" onmousedown="event.stopPropagation();">
            <b>rsid:</b> {point.label}
          </p>
          <a href="https://www.ncbi.nlm.nih.gov/snp/?term={point.label}" target="_blank" rel="noopener noreferrer" style="user-select:text;" onmousedown="event.stopPropagation();" class="text-blue-main hover:text-blue-dark">
            <p><b>DBSNP link</b></p></a>
        </div>`,
      useHTML: true,
      stickOnContact: true,
    },
    series: [
      {
        data: variants,
        turboThreshold: 0,
        boostThreshold: 1,
        marker: {
          radius: 3,
        },
        allowPointSelect: true,
        accessibility: {
          enabled: false,
        },
      },
    ],
  };
  return (
    <div>
      <HighchartsReact highcharts={Highcharts} options={options} />
      <div className="flex flex-row justify-center mb-4">
        <Legend colorMap={colorMap} />
      </div>
      <Docs />
    </div>
  );
}
