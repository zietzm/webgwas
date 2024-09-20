import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import HighchartsBoost from "highcharts/modules/boost";
import { VariantPvalue } from "../lib/api";
HighchartsBoost(Highcharts);

export default function ManhattanPlot({ data }: { data: VariantPvalue[] }) {
  const chrData: Map<
    string,
    Array<{ label: string; bp: number; p: number }>
  > = new Map();
  data.forEach((d) => {
    const [chr, bpStr] = d.variant_id.split(":");
    const bp = parseInt(bpStr, 10);
    const y = d.pvalue;
    if (!chrData.has(chr)) {
      chrData.set(chr, []);
    }
    chrData.get(chr)!.push({ label: d.variant_id, bp: bp, p: y });
  });

  // Get unique chromosomes and sort them
  const chromosomes = Array.from(chrData.keys()).toSorted((a, b) => {
    const chrA = a === "X" ? 23 : a === "Y" ? 24 : parseInt(a);
    const chrB = b === "X" ? 23 : b === "Y" ? 24 : parseInt(b);
    return chrA - chrB;
  });

  // Calculate maximum position for each chromosome
  const chrInfo = chromosomes.map((chr) => {
    const positions = chrData.get(chr)!.map((d) => d.bp);
    const maxPos = Math.max(...positions);
    return {
      chr: chr,
      maxPos: maxPos,
    };
  });

  // Calculate cumulative offsets for chromosome positions
  const chrOffsets: Map<string, number> = new Map();
  let cumulativeOffset = 0;
  chrInfo.forEach((info) => {
    chrOffsets.set(info.chr, cumulativeOffset);
    cumulativeOffset += info.maxPos;
  });

  // Assign colors to chromosomes
  const defaultColors = Highcharts.getOptions().colors;
  if (defaultColors === undefined) {
    throw new Error("Highcharts colors not defined");
  }
  const chrColorMap: Map<string, any> = new Map();
  chromosomes.forEach((chr, index) => {
    chrColorMap.set(chr, defaultColors[index % defaultColors.length]);
  });

  // Adjust positions and prepare data points
  const finalData: Array<{
    x: number;
    y: number;
    color: any;
    label: string;
  }> = [];
  chromosomes.forEach((chr) => {
    const offset = chrOffsets.get(chr)!;
    const color = chrColorMap.get(chr)!;
    const points = chrData.get(chr)!.map((point) => {
      return {
        x: point.bp + offset,
        y: point.p,
        color: color,
        label: point.label,
      };
    });
    finalData.push(...points);
  });

  // Calculate tick positions for the x-axis
  const chrTicks = chromosomes.map((chr) => {
    const offset = chrOffsets.get(chr)!;
    const positions = chrData.get(chr)!.map((d) => d.bp);
    const minPos = Math.min(...positions);
    const maxPos = Math.max(...positions);
    const midPos = offset + (minPos + maxPos) / 2;
    return {
      value: midPos,
      label: chr,
    };
  });

  const tickPositionToChr: Map<number, string> = new Map();
  chrTicks.forEach((tick) => {
    tickPositionToChr.set(tick.value, tick.label);
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
      tickPositions: chrTicks.map((tick) => tick.value),
      labels: {
        formatter: function (): any {
          return tickPositionToChr.get(this.value);
        },
      },
      showLastLabel: true,
    },
    yAxis: {
      minPadding: 0,
      maxPadding: 0,
      title: {
        text: "Negative log 10 p-value",
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
      pointFormat: "{point.label}",
    },
    series: [
      {
        data: finalData,
        turboThreshold: 0,
        boostThreshold: 1,
        marker: {
          radius: 2,
        },
      },
    ],
  };
  return (
    <div>
      <HighchartsReact highcharts={Highcharts} options={options} />
    </div>
  );
}
