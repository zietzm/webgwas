"use client";

import { useState } from "react";

import {
  Cohort,
  operators,
  PhenotypeNode,
  PhenotypeSummary,
} from "@/app/lib/types";
import MainArea from "@/components/main_area";
import CohortSelector from "@/components/cohort_selector";
import Accordion from "@/components/accordion";
import { GwasHandlerDoValidate } from "@/components/gwas_handler";
import FitQuality from "@/components/fit_quality";
import GwasResults from "@/components/gwas_results";
import { PvaluesResult } from "@/app/lib/api";

import TreePhenotypeBuilder from "./tree_phenotype_builder";

const startPhenotype: PhenotypeNode = {
  id: 0,
  data: operators[0],
  children: [],
};

export default function TreePage() {
  const [cohort, setCohort] = useState<Cohort | null>(null);
  const [phenotype, setPhenotype] = useState<PhenotypeNode>(startPhenotype);
  const [isMutable, setIsMutable] = useState<boolean>(true);
  const [fitQuality, setFitQuality] = useState<PhenotypeSummary | null>(null);
  const [results, setResults] = useState<PvaluesResult | null>(null);

  function handleReset() {
    setCohort(null);
    setPhenotype(startPhenotype);
    setIsMutable(true);
    setFitQuality(null);
    setResults(null);
  }

  return (
    <div className="flex flex-col gap-4">
      <HeaderInformation />
      <Accordion title="Usage Instructions">
        <TreePhenotypeBuilderUsage />
      </Accordion>
      <MainArea>
        <h2>Select Cohort</h2>
        <CohortSelector
          selectedCohort={cohort}
          setSelectedCohort={setCohort}
          isMutable={isMutable}
        />
        {cohort && (
          <>
            <h2>Build GWAS phenotype tree</h2>
            <TreePhenotypeBuilder
              cohort={cohort}
              phenotype={phenotype}
              setPhenotype={setPhenotype}
              isMutable={isMutable}
            />
          </>
        )}
        {cohort && phenotype.children.length > 0 && (
          <GwasHandlerDoValidate
            cohort={cohort}
            phenotype={{
              type: "tree",
              value: phenotype,
            }}
            setIsMutable={setIsMutable}
            setFitQuality={setFitQuality}
            setResults={setResults}
            handleReset={handleReset}
          />
        )}
        {fitQuality && (
          <>
            <h2>Approximating your phenotype</h2>
            <FitQuality data={fitQuality} />
          </>
        )}
        {results && (
          <>
            <h2>Summary of results: Manhattan plot</h2>
            <GwasResults results={results} />
          </>
        )}
      </MainArea>
    </div>
  );
}

function HeaderInformation() {
  return (
    <div className="my-6 text-gray-600 dark:text-gray-400">
      <p className="text-3xl font-bold text-center mb-8 text-blue-dark">
        Advanced, tree-based builder for phenotype definitions
      </p>
      <p>
        This is the most flexible builder for defining phenotypes. Unlike the
        simple builder, this includes quantitative features and allows a larger
        set of operators.
      </p>
    </div>
  );
}

function TreePhenotypeBuilderUsage() {
  return (
    <div className="flex flex-col gap-2">
      <p>
        To get started, select a cohort from the buttons below. Then define the
        phenotype that interests you by searching for fields in the search box
        below.
      </p>
      <p>
        As an example, suppose we are interested hypertensive diabetes (diabetes
        and hypertension). To build this phenotype, we would add both diabetes
        (E11) and hypertension (I10) as children.
      </p>
      <p>The resulting phenotype would look like this:</p>
      <div className="flex flex-row">
        <div className="basis-full md:basis-3/5">
          <img
            src="/tree_example.webp"
            alt="Usage example for the tree builder"
          />
        </div>
      </div>
      <p>
        Once built, you can validate your phenotype and run the GWAS. Our server
        will then start the GWAS calculation and display the status of the job.
        Once complete, a download link to your results will appear.
      </p>
    </div>
  );
}
