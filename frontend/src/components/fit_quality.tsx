import { PhenotypeSummary } from "@/app/lib/types";
import React from "react";
import QualityGauge from "@/app/ui/QualityGauge";
import PhenotypeScatterPlots from "@/app/ui/PhenotypeSummary";
import Accordion from "@/components/accordion";

export default function FitQuality({ data }: { data: PhenotypeSummary }) {
  return (
    <div className="flex flex-col">
      <div className="flex flex-wrap justify-center md:justify-between gap-4">
        <div className="basis-1/2 grow">
          <TextInfo />
        </div>
        <div>
          <QualityGauge quality={data.rsquared!} />
        </div>
      </div>
      <Accordion title="More information">
        <PhenotypeScatterPlots data={data} />
      </Accordion>
    </div>
  );
}

function TextInfo(): React.ReactElement {
  return (
    <div className="flex flex-col gap-4">
      <p>
        WebGWAS is much faster than a traditional GWAS thanks to an
        approximation we use under the hood.
      </p>
      <p>
        How well we can approximate your phenotype determines how accurate the
        GWAS will be.
      </p>
      <p>
        The approximation can be evaluated using the regression R<sup>2</sup>,
        shown in the gauge.
      </p>
    </div>
  );
}
