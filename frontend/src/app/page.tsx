"use client";

import { useState } from "react";
import SimplePhenotypeBuilder, {
  simplePhenotypeBuilderUsage,
} from "@/app/ui/SimplePhenotype";
import { ChevronDown, ChevronUp } from "lucide-react";

function headerInformation() {
  return (
    <div className="mb-6 text-gray-600 dark:text-gray-400">
      <p className="text-xl font-bold text-center mb-8 text-blue-dark">
        Instant, free genome-wide association studies (GWAS) on arbitrary
        phenotypes
      </p>
      <p>
        Using{" "}
        <a
          href="/about"
          className="underline text-jet font-medium hover:text-blue-dark visited:text-purple-600"
        >
          a new approximation method
        </a>
        , WebGWAS provides GWAS summary statistics for arbitrary phenotype
        definitions. Whereas the{" "}
        <a
          href="https://pan.ukbb.broadinstitute.org/"
          className="underline text-jet font-medium hover:text-blue-dark visited:text-purple-600"
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
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div>
      <h2>
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center justify-between w-full p-5 font-medium rtl:text-right bg-gray-100 text-gray-500 border border-gray-200 dark:border-gray-700 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-800 gap-3"
        >
          <span>Usage Instructions</span>
          {isOpen ? (
            <ChevronUp className="font-bold w-5 h-5 shrink-0" />
          ) : (
            <ChevronDown className="font-bold w-5 h-5 shrink-0" />
          )}
        </button>
      </h2>
      {isOpen && simplePhenotypeBuilderUsage()}
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
