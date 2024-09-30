import React from "react";
import { PhenotypeSummary } from "../lib/types";
import PhenotypePlot from "./PhenotypePlot";
import PhenotypeFitPlot from "./PhenotypeFitPlot";

interface PhenotypeScatterPlotsProps {
  data: PhenotypeSummary;
}

export function SummaryDocumentation() {
  return (
    <div className="my-2">
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        The left figure compares the true and approximated phenotype values. The
        quality of this approximation determines the fidelity of the GWAS. The
        right plot shows the relationship between phenotype fits and GWAS
        summary statistics. The red line shows the fit quality of the phenotype
        you defined.
      </p>
      <p className="text-gray-600 dark:text-gray-400">
        Note: No point in the phenotype plot has fewer than 10 data points. This
        is because we anonymized the data using k-anonymity with k=10.
      </p>
    </div>
  );
}

const PhenotypeScatterPlots: React.FC<PhenotypeScatterPlotsProps> = ({
  data,
}) => {
  let phenotypeData: number[][] = data.phenotype_values.map((p) => [p.t, p.a]);
  let fitQualityData: number[][] = data.fit_quality_reference.map((p) => [
    p.p,
    p.g,
  ]);

  return (
    <div className="flex flex-col space-y-4 mt-4">
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
            rsquared={data.rsquared}
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
