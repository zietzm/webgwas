import PhenotypeBuilder from "@/app/ui/PhenotypeBuilder";
import Image from "next/image";

export default function Home() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-8 text-indigo-800">WebGWAS Phenotype Builder</h1>
      <p className="mb-2 text-gray-600 dark:text-gray-400">A tool for running Genome-Wide Association Studies (GWAS) on arbitrary phenotypes.</p>
      <p className="mb-2 text-gray-600 dark:text-gray-400">Use the <a href="https://pan.ukbb.broadinstitute.org/" className="underline text-blue-600 hover:text-blue-800 visited:text-purple-600">Pan-UK Biobank</a> to download GWAS summary statistics for a simple phenotype (e.g. hypertension defined as ICD-10: I10). If you want GWAS summary statistics for a more complicated phenotype definition (e.g. hypertension defined as either ICD-10 I10 or Phecode 401), then WebGWAS is the appropriate tool!</p>
      <div id="accordion-open" data-accordion="open">
        <h2 id="accordion-collapse-heading-3">
          <button type="button" className="flex items-center justify-between w-full p-5 font-medium rtl:text-right text-gray-500 border border-gray-200 focus:ring-4 focus:ring-gray-200 dark:focus:ring-gray-800 dark:border-gray-700 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 gap-3" data-accordion-target="#accordion-collapse-body-3" aria-expanded="true" aria-controls="accordion-collapse-body-3">
            <span>Usage Instructions</span>
            <svg data-accordion-icon className="w-3 h-3 rotate-180 shrink-0" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 10 6">
              <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5 5 1 1 5"/>
            </svg>
          </button>
        </h2>
        <div id="accordion-collapse-body-3" className="hidden" aria-labelledby="accordion-collapse-heading-3">
          <div className="p-5 border border-t-0 border-gray-200 dark:border-gray-700">
            <p className="mb-2 text-gray-600 dark:text-gray-400">To get started, select a cohort from the buttons below. Then define the phenotype that interests you by clicking the "+" button and selecting the type of node you want to add.</p>
            <p className="mb-2 text-gray-600 dark:text-gray-400">As an example, suppose we are interested hypertensive diabetes (diabetes and hypertension). To build this phenotype, we would select first the "AND" operator, then add both diabetes (E11) and hypertension (I10) as children.</p>
            <p className="mb-2 text-gray-600 dark:text-gray-400">The resulting phenotype would look like this:</p>
            <Image src={"/hypertensive_diabetes_example.png"} alt="Example phenotype" width={250} height={131} />
            <p className="mb-2 text-gray-600 dark:text-gray-400">Once built, you can validate your phenotype and run the GWAS. Our server will then start the GWAS calculation and display the status of the job. Once complete, a download link to your results will appear.</p>
    </div>
  </div>
      </div>
      <br />
      <PhenotypeBuilder />
    </div>
  );
}
