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

const CustomTooltip = ({ active, payload }: any) => {
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

const PhenotypeScatterPlots: React.FC<PhenotypeScatterPlotsProps> = ({
  data,
}) => {
  const { phenotypes, rsquared, fit_quality } = data;

  const minValue = Math.min(...phenotypes.map((p) => Math.min(p.t, p.a)));
  const maxValue = Math.max(...phenotypes.map((p) => Math.max(p.t, p.a)));

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
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={<CustomTooltip />}
              />
              <Scatter
                name="Phenotypes"
                data={phenotypes}
                fill="#8884d8"
                fillOpacity={0.5}
              />
              <ReferenceLine
                stroke="grey"
                strokeDasharray="3 3"
                segment={[
                  { x: minValue, y: minValue },
                  { x: maxValue, y: maxValue },
                ]}
                ifOverflow="hidden"
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
                  value: "Phenotype fit (R^2)",
                  position: "bottom",
                  offset: 0,
                }}
              />
              <YAxis
                type="number"
                dataKey="g"
                tick={{ fontSize: 12 }}
                label={{
                  value: "GWAS log p-value fit (R^2)",
                  angle: -90,
                  position: "left",
                  offset: 12,
                  style: { textAnchor: "middle" },
                }}
              />
              <ZAxis type="number" dataKey="n" />
              <ReferenceLine
                x={rsquared}
                stroke="red"
                r={10}
                label={`R^2 = ${rsquared.toFixed(2)}`}
              />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={<CustomTooltip />}
              />
              <ReferenceLine
                stroke="grey"
                strokeDasharray="3 3"
                segment={[
                  { x: minValue, y: minValue },
                  { x: maxValue, y: maxValue },
                ]}
                ifOverflow="hidden"
              />
              <Scatter
                name="Fit Quality"
                data={fit_quality}
                fill="#82ca9d"
                fillOpacity={0.5}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default PhenotypeScatterPlots;
