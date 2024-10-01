import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import highchartsAccessibility from "highcharts/modules/accessibility";
import HighchartsBoost from "highcharts/modules/boost";
if (typeof Highcharts === "object") {
  highchartsAccessibility(Highcharts);
  HighchartsBoost(Highcharts);
}

export default function PhenotypePlot({
  data,
  xlab,
  ylab,
  title,
}: {
  data: number[][];
  xlab: string;
  ylab: string;
  title: string;
}) {
  const maxVal = Math.max(...data.map((d) => Math.max(...d)));
  const minVal = Math.min(...data.map((d) => Math.min(...d)));
  const dataRange = maxVal - minVal;
  const displayMinVal = minVal - 0.02 * dataRange;
  const displayMaxVal = maxVal + 0.02 * dataRange;

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
      text: title,
    },
    xAxis: {
      min: displayMinVal,
      max: displayMaxVal,
      gridLineWidth: 1,
      minPadding: 0,
      maxPadding: 0,
      title: {
        text: xlab,
        style: {
          fontSize: "17px",
          color: "rgb(75 85 99)",
        },
      },
      showLastLabel: true,
    },
    yAxis: {
      min: displayMinVal,
      max: displayMaxVal,
      gridLineWidth: 1,
      minPadding: 0,
      maxPadding: 0,
      title: {
        text: ylab,
        style: {
          fontSize: "17px",
          color: "rgb(75 85 99)",
        },
      },
      showLastLabel: true,
    },
    legend: {
      enabled: false,
    },
    tooltip: {
      enabled: false,
    },
    series: [
      {
        data: [
          [0, 0],
          [1, 1],
        ],
        type: "line",
        color: "grey",
        dashStyle: "dash",
        lineWidth: 1,
        opacity: 0.5,
        animation: false,
        zIndex: 0,
      },
      {
        data: data,
        type: "scatter",
        turboThreshold: 0,
        boostThreshold: 1,
        marker: {
          symbol: "circle",
          radius: 5,
          fillColor: "rgba(89, 130, 179, .1)",
        },
        opacity: 0,
        zIndex: 1,
      },
    ],
  };
  return (
    <div>
      <HighchartsReact highcharts={Highcharts} options={options} />
    </div>
  );
}
