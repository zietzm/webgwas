import React from "react";
import QualityGauge from "./QualityGauge";
import PhenotypeScatterPlots from "./PhenotypeSummary";
import { PhenotypeSummary } from "../lib/types";
import {
  Accordion,
  AccordionTrigger,
  AccordionContent,
  AccordionItem,
} from "@/components/ui/accordion";
export default function QualityInformation({
  data,
}: {
  data: PhenotypeSummary;
}) {
  return (
    <div className="flex flex-col">
      <div className="flex flex-wrap justify-center md:justify-between gap-4 mt-4">
        <div className="basis-1/2 grow">
          <div className="text-start text-2xl font-semibold">
            Approximating your phenotype
          </div>
          <div className="mt-2 text-start">
            WebGWAS is much faster than a traditional GWAS thanks to an
            approximation we use under the hood.
          </div>
          <div className="mt-2 text-start">
            How well we can approximate your phenotype determines how accurate
            the GWAS will be.
          </div>
          <div className="mt-2 text-start">
            The approximation can be evaluated using the regression R
            <sup>2</sup>, shown in the gauge.
          </div>
        </div>
        <div className="">
          <QualityGauge quality={data.rsquared!} />
        </div>
      </div>
      <div className="">
        <Accordion type="single" collapsible>
          <AccordionItem value="item-1">
            <AccordionTrigger>More information</AccordionTrigger>
            <AccordionContent>
              <PhenotypeScatterPlots data={data} />
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </div>
  );
}
