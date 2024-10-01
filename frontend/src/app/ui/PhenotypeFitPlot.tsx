import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import highchartsAccessibility from "highcharts/modules/accessibility";
import HighchartsBoost from "highcharts/modules/boost";
if (typeof Highcharts === "object") {
  highchartsAccessibility(Highcharts);
  HighchartsBoost(Highcharts);
}

export default function PhenotypeFitPlot({
  data,
  rsquared,
  xlab,
  ylab,
  title,
}: {
  data: number[][];
  rsquared: number | null;
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
      min: -0.02,
      max: 1.02,
      gridLineWidth: 1,
      minPadding: 0,
      maxPadding: 0,
      title: {
        text: xlab,
        useHTML: true,
        style: {
          fontSize: "17px",
        },
      },
      showLastLabel: true,
      plotLines: [
        {
          color: "#FF0000",
          width: 2,
          value: rsquared,
          dashStyle: "dash",
          label: {
            text:
              rsquared === null
                ? "R<sup>2</sup> = Not Available"
                : "R<sup>2</sup> = " + rsquared.toFixed(2),
            useHTML: true,
            rotation: 0,
            verticalAlign: "top",
            y: 30,
            textAlign:
              rsquared === null ? "center" : rsquared > 0.5 ? "right" : "left",
            x: rsquared === null ? 0 : rsquared > 0.5 ? -10 : 10,
            style: {
              fontSize: "17px",
            },
          },
        },
      ],
    },
    yAxis: {
      min: 0,
      max: 1,
      gridLineWidth: 1,
      minPadding: 0,
      maxPadding: 0,
      title: {
        text: ylab,
        useHTML: true,
        style: {
          fontSize: "17px",
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
    annotations: [
      {
        label: {
          text:
            "R<sup>2</sup> = " +
            (rsquared === null ? "Not Available" : rsquared.toFixed(2)),
          x: 0.5,
          y: 0.5,
        },
        type: "label",
      },
    ],
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
          fillColor: "rgba(64, 98, 140, .1)",
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
