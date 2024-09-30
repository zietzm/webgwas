import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import highchartsAccessibility from "highcharts/modules/accessibility";
import highchartsExporting from "highcharts/modules/exporting";
import HighchartsBoost from "highcharts/modules/boost";
import { PvaluesResult } from "../lib/api";
if (typeof Highcharts === "object") {
  highchartsAccessibility(Highcharts);
  HighchartsBoost(Highcharts);
  highchartsExporting(Highcharts);
  Highcharts.AST.allowedTags.push("input");
  Highcharts.AST.allowedAttributes.push("rel");
  Highcharts.AST.allowedAttributes.push("onmousedown");
}

export function Docs() {
  return (
    <p className="text-gray-600 dark:text-gray-400 mb-2">
      For speed, the plot above shows only variants with p &lt; 0.05. Unfiltered
      results are available in the download.
    </p>
  );
}

export default function ManhattanPlot({ data }: { data: PvaluesResult }) {
  const colors = ["#1d2f6f", "#8390fa"];

  const chrPositions = data.chromosome_positions.map((chr) => chr.midpoint);

  const chrIdxToLabel: Map<number, string> = new Map();
  data.chromosome_positions.forEach((chr) => {
    chrIdxToLabel.set(chr.midpoint, chr.chromosome);
  });

  const chrToColor: Map<string, string> = new Map();
  data.chromosome_positions.forEach((chr, index) => {
    chrToColor.set(chr.chromosome, colors[index % colors.length]);
  });

  const variants = data.pvalues.map((p) => {
    return {
      x: p.index,
      y: p.pvalue,
      color: chrToColor.get(p.chromosome)!,
      label: p.label,
    };
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
      text: "Manhattan Plot",
    },
    xAxis: {
      gridLineWidth: 1,
      title: {
        text: "Variant position",
      },
      tickPositions: chrPositions,
      labels: {
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
          <p style="user-select:text; cursor:text;" onmousedown="event.stopPropagation();"><b>rsid:</b> {point.label}</p>
          <a href="https://www.ncbi.nlm.nih.gov/snp/?term={point.label}" target="_blank" rel="noopener noreferrer" style="user-select:text; cursor:pointer;" onmousedown="event.stopPropagation();" class="text-blue-600 hover:text-blue-800 visited:text-purple-600"><p><b>DBSNP link</b></p></a>
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
          radius: 2,
        },
        allowPointSelect: true,
        accessibility: {
          enabled: true,
          description: "RSid of the variant",
        },
      },
    ],
  };
  return (
    <div>
      <HighchartsReact highcharts={Highcharts} options={options} />
      <Docs />
    </div>
  );
}
