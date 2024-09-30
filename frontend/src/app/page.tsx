"use client";

import SimplePhenotypeBuilder, {
  simplePhenotypeBuilderUsage,
} from "@/app/ui/SimplePhenotype";

function headerInformation() {
  return (
    <div className="mb-6 text-gray-600 dark:text-gray-400">
      <p className="text-xl font-bold text-center mb-8 text-indigo-800">
        Instant, free genome-wide association studies (GWAS) on arbitrary
        phenotypes
      </p>
      <p>
        Using{" "}
        <a
          href="/about"
          className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
        >
          a new approximation method
        </a>
        , WebGWAS provides GWAS summary statistics for arbitrary phenotype
        definitions. Whereas the{" "}
        <a
          href="https://pan.ukbb.broadinstitute.org/"
          className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600"
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

function accordionUsage() {
  return (
    <div id="accordion-collapse" data-accordion="collapse">
      <h2 id="accordion-collapse-heading-3">
        <button
          type="button"
          className="flex items-center justify-between w-full p-5 font-medium rtl:text-right text-gray-500 border border-gray-200 focus:ring-4 focus:ring-gray-200 dark:focus:ring-gray-800 dark:border-gray-700 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 gap-3"
          data-accordion-target="#accordion-collapse-body-3"
          aria-expanded="false"
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
      {headerInformation()}
      {accordionUsage()}
      <br />
      <SimplePhenotypeBuilder />
    </div>
  );
}
