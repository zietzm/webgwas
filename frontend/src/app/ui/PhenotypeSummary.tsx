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
  ReferenceDot,
} from "recharts";
import { PhenotypeSummary } from "../lib/types";

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
        <p className="">{`Phenotype fit (R^2) : ${payload[0].value}`}</p>
        <p className="">{`GWAS log p-value fit (R^2) : ${payload[1].value}`}</p>
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
  const { phenotypes, rsquared, fit_quality } = data;

  const minX = Math.min(...phenotypes.map((p) => p.t));
  const minY = Math.min(...phenotypes.map((p) => p.a));
  const maxX = Math.max(...phenotypes.map((p) => p.t));
  const maxY = Math.max(...phenotypes.map((p) => p.a));
  const xMin = Math.min(minX, minY);
  const xMax = Math.max(maxX, maxY);

  return (
    <div className="flex flex-col space-y-4">
      <h2 className="text-xl font-bold">Summary of the phenotype</h2>
      <div className="flex flex-row space-x-2">
        <div className="w-1/2">
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid />
              <XAxis
                type="number"
                dataKey="t"
                tick={{ fontSize: 12 }}
                label={{
                  value: "True phenotype",
                  position: "bottom",
                  offset: 0,
                }}
              />
              <YAxis
                type="number"
                dataKey="a"
                tick={{ fontSize: 12 }}
                label={{
                  value: "Approximate phenotype",
                  angle: -90,
                  position: "left",
                  offset: 12,
                  style: { textAnchor: "middle" },
                }}
              />
              <ZAxis type="number" dataKey="n" />
              <Scatter
                name="Phenotypes"
                data={phenotypes}
                fill="#8884d8"
                fillOpacity={0.5}
              />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={<PhenotypeTooltip />}
              />
              <ReferenceLine
                stroke="grey"
                strokeDasharray="3 3"
                segment={[
                  { x: xMin, y: xMin },
                  { x: xMax, y: xMax },
                ]}
                ifOverflow="hidden"
              />
              <ReferenceDot
                x={minX}
                y={maxY}
                r={0}
                label={
                  <CustomLabel
                    dy={30}
                    value={rsquared.toFixed(2)}
                    fill="#666"
                  />
                }
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="w-1/2">
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid />
              <XAxis
                type="number"
                dataKey="p"
                tick={{ fontSize: 12 }}
                label={{
                  value: "Phenotype fit",
                  position: "bottom",
                  offset: 0,
                }}
              />
              <YAxis
                type="number"
                dataKey="g"
                tick={{ fontSize: 12 }}
                label={{
                  value: "GWAS statistics fit",
                  angle: -90,
                  position: "left",
                  offset: 12,
                  style: { textAnchor: "middle" },
                }}
              />
              <ZAxis type="number" dataKey="n" />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={<FitQualityTooltip />}
              />
              <ReferenceLine
                stroke="grey"
                strokeDasharray="3 3"
                segment={[
                  { x: 0, y: 0 },
                  { x: 1, y: 1 },
                ]}
                ifOverflow="hidden"
              />
              <Scatter
                name="Fit Quality"
                data={fit_quality}
                fill="#82ca9d"
                fillOpacity={0.5}
              />
              <ReferenceLine
                x={rsquared}
                stroke="red"
                r={10}
                label={
                  <CustomLabel
                    value={rsquared.toFixed(2)}
                    dy={100}
                    dx={rsquared > 0.5 ? -35 : 35}
                    fill="red"
                  />
                }
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
      <SummaryDocumentation />
    </div>
  );
};

export default PhenotypeScatterPlots;
