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
      min: 0,
      max: 1,
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
      min: 0,
      max: 1,
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
