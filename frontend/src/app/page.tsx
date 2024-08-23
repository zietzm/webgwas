"use client";

import SimplePhenotypeBuilder, {
  simplePhenotypeBuilderUsage,
} from "@/app/ui/SimplePhenotype";

function headerInformation() {
  return (
    <div className="mb-6 text-gray-600 dark:text-gray-400">
      <p>
        A tool for running Genome-Wide Association Studies (GWAS) on arbitrary
        phenotypes.
      </p>
      <p>
        Use the{" "}
        <a
          href="https://pan.ukbb.broadinstitute.org/"
          className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
        >
          Pan-UK Biobank
        </a>{" "}
        to download GWAS summary statistics for a simple phenotype (e.g.
        hypertension defined as ICD-10: I10). If you want GWAS summary
        statistics for a more complicated phenotype definition (e.g.
        hypertension defined as either ICD-10 I10 or Phecode 401), then WebGWAS
        is the appropriate tool!
      </p>
    </div>
  );
}

function accordionUsage() {
  return (
    <div id="accordion-open" data-accordion="open">
      <h2 id="accordion-collapse-heading-3">
        <button
          type="button"
          className="flex items-center justify-between w-full p-5 font-medium rtl:text-right text-gray-500 border border-gray-200 focus:ring-4 focus:ring-gray-200 dark:focus:ring-gray-800 dark:border-gray-700 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 gap-3"
          data-accordion-target="#accordion-collapse-body-3"
          aria-expanded="true"
          aria-controls="accordion-collapse-body-3"
        >
          <span>Usage Instructions</span>
          <svg
            data-accordion-icon
            className="w-3 h-3 rotate-180 shrink-0"
            aria-hidden="true"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 10 6"
          >
            <path
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M9 5 5 1 1 5"
            />
          </svg>
        </button>
      </h2>
      <div
        id="accordion-collapse-body-3"
        className="hidden"
        aria-labelledby="accordion-collapse-heading-3"
      >
        {simplePhenotypeBuilderUsage()}
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-8 text-indigo-800">
        WebGWAS Phenotype Builder
      </h1>
      {headerInformation()}
      {accordionUsage()}
      <br />
      <SimplePhenotypeBuilder />
    </div>
  );
}
