"use client";

import SimplePhenotypeBuilder, {
  SimplePhenotypeBuilderUsage,
} from "@/app/ui/SimplePhenotype";
import Accordion from "@/components/accordion";

function headerInformation() {
  return (
    <div className="mb-6 text-gray-600 dark:text-gray-400">
      <p className="text-3xl font-bold text-center mb-8 text-blue-dark">
        Instant, free genome-wide association studies (GWAS) on arbitrary
        phenotypes
      </p>
      <p>
        Using{" "}
        <a
          href="/about"
          className="underline text-jet font-medium hover:text-blue-dark visited:text-purple-800"
        >
          a new approximation method
        </a>
        , WebGWAS provides GWAS summary statistics for arbitrary phenotype
        definitions. Whereas the{" "}
        <a
          href="https://pan.ukbb.broadinstitute.org/"
          className="underline text-jet font-medium hover:text-blue-dark visited:text-purple-800"
        >
          Pan-UK Biobank
        </a>{" "}
        provides pre-computed results for individual phenotypes (e.g.
        hypertension defined as ICD-10: I10), WebGWAS lets you study any
        phenotypes that interest you.
      </p>
    </div>
  );
}

export default function Home() {
  return (
    <div className="max-w-4xl mx-auto">
      {headerInformation()}
      <Accordion title="Usage Instructions">
        <SimplePhenotypeBuilderUsage />
      </Accordion>
      <br />
      <SimplePhenotypeBuilder />
    </div>
  );
}
