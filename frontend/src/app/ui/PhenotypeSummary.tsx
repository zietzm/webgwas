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
        <b>
          *<i>GWAS summary statistic fit</i>
        </b>{" "}
        is the R<sup>2</sup> of negative log<sub>10</sub> p-values between
        approximated GWAS on anonymized data and true GWAS on original data.
      </p>
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        The red line shows the fit quality of your phenotype.
      </p>
      <p className="text-gray-600 dark:text-gray-400">
        Data are anonymized using k-anonymity with k=10 to ensure that privacy
        is protected.
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
      <h2 className="text-xl font-bold">
        Phenotype QC: Quality of the approximation
      </h2>
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        WebGWAS provides results quickly by approximating your phenotype. How
        well it can be approximated determines the quality of the GWAS. We use a
        linear regression for this approximation.
      </p>
      <div className="flex flex-row space-x-2">
        <div className="w-1/2">
          <PhenotypePlot
            data={phenotypeData}
            xlab="True phenotype"
            ylab="Approximated phenotype"
            title="True vs predicted phenotype"
          />
        </div>
        <div className="w-1/2">
          <PhenotypeFitPlot
            data={fitQualityData}
            rsquared={data.rsquared}
            xlab="Phenotype fit (R<sup>2</sup> true vs predicted)"
            ylab="GWAS statistics fit <b>*</b>"
            title="Fit quality vs GWAS fidelity"
          />
        </div>
      </div>
      <SummaryDocumentation />
    </div>
  );
};

export default PhenotypeScatterPlots;
