import React from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ZAxis,
} from "recharts";
import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import HighchartsBoost from "highcharts/modules/boost";
import { PhenotypeSummary } from "../lib/types";
import PhenotypePlot from "./PhenotypePlot";
import PhenotypeFitPlot from "./PhenotypeFitPlot";
HighchartsBoost(Highcharts);

interface PhenotypeScatterPlotsProps {
  data: PhenotypeSummary;
}

export function SummaryDocumentation() {
  return (
    <div className="p-5">
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        The left figure compares the true and approximated phenotype values. The
        quality of this approximation determines the fidelity of the GWAS. The
        right plot shows the relationship between phenotype fits and GWAS
        summary statistics. The red line shows the fit quality of the phenotype
        you defined.
      </p>
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        Note: No point in the phenotype plot has fewer than 10 data points. This
        is because we anonymized the data using k-anonymity with k=10.
      </p>
    </div>
  );
}

const PhenotypeTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white opacity-75 rounded-lg shadow-lg p-4 border border-gray-200">
        <p className="">{`True : ${payload[0].value}`}</p>
        <p className="">{`Approx : ${payload[1].value}`}</p>
        <p className="">{`N : ${payload[2].value}`}</p>
      </div>
    );
  }
  return null;
};

const FitQualityTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white opacity-75 rounded-lg shadow-lg p-4 border border-gray-200">
        <p>
          Phenotype fit (R<sup>2</sup>): {payload[0].value}
        </p>
        <p>
          GWAS log p-value fit (R<sup>2</sup>): {payload[1].value}
        </p>
      </div>
    );
  }
  return null;
};

const CustomLabel = ({ viewBox, value, dx, dy, fill }: any) => {
  return (
    <g>
      <text
        x={viewBox.x}
        y={viewBox.y}
        fill={fill}
        fontSize="14"
        textAnchor="middle"
        dx={dx}
        dy={dy}
      >
        R
        <tspan fontSize="10" baselineShift="super">
          2
        </tspan>
        = {value}
      </text>
    </g>
  );
};

const PhenotypeScatterPlots: React.FC<PhenotypeScatterPlotsProps> = ({
  data,
}) => {
  const { phenotype_values, rsquared, fit_quality_reference } = data;

  let phenotypeData: number[][] = phenotype_values.map((p) => [p.t, p.a]);
  let fitQualityData: number[][] = fit_quality_reference.map((p) => [p.p, p.g]);

  return (
    <div className="flex flex-col space-y-4">
      <h2 className="text-xl font-bold">Summary of the phenotype</h2>
      <div className="flex flex-row space-x-2">
        <div className="w-1/2">
          <PhenotypePlot
            data={phenotypeData}
            xlab="True phenotype"
            ylab="Approximate phenotype"
            title="Phenotypes"
          />
        </div>
        <div className="w-1/2">
          <PhenotypeFitPlot
            data={fitQualityData}
            rsquared={rsquared}
            xlab="Phenotype fit"
            ylab="GWAS statistics fit"
            title="Fit quality"
          />
        </div>
      </div>
      <SummaryDocumentation />
    </div>
  );
};

export default PhenotypeScatterPlots;
