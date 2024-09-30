export default function FAQ() {
  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold mb-6 text-blue-dark">
        Frequently Asked Questions
      </h1>
      <div className="prose prose-jet">
        <h2 className="text-xl font-semibold mb-4">What is a GWAS?</h2>
        <p>
          A GWAS is a statistical analysis that compares the frequency of a
          genetic variant (SNP) in a cohort of individuals to the frequency of
          that variant in a reference population. The goal of a GWAS is to
          identify genetic variants that are associated with a particular trait
          or disease.
        </p>
        <br />
        <h2 className="text-xl font-semibold mb-4">What is a phenotype?</h2>
        <p>
          A phenotype is an observable characteristic of an individual that
          results from the interaction of genetic variants and environmental
          factors. It is a way to describe the phenotypic expression of a trait
          or disease, and it can be used to identify genetic variants that are
          associated with a particular trait or disease. In observational
          studies, the definition of a phenotype is important, and is often made
          up of fields from electronic health records, measurements, or other
          observations.
        </p>
        <br />
        <h2 className="text-xl font-semibold mb-4">How does WebGWAS work?</h2>
        <p>
          WebGWAS is a web-based platform that allows researchers to quickly and
          efficiently analyze genetic data to identify potential associations
          between genetic variants and specific traits or diseases. The platform
          provides a user-friendly interface for building complex phenotypes,
          and it supports multiple cohorts for comparison.
        </p>
        <br />
        <h2 className="text-xl font-semibold mb-4">How do I use WebGWAS?</h2>
        <p>
          WebGWAS performs an approximate GWAS calculation by combining the
          results of multiple GWAS calculations. You can start building your
          phenotype by selecting a cohort from the dropdown menu. You can then
          add nodes to your phenotype by clicking on the &ldquo;+&rdquo; button
          and selecting the type of node you want to add. You can also remove
          nodes by clicking on the &ldquo;-&rdquo; button next to the node. Once
          you have built your phenotype, you can validate it by clicking the
          &ldquo;Validate Phenotype&rdquo; button. If the phenotype is valid,
          you can run the GWAS by clicking the &ldquo;Run GWAS&rdquo; button.
          The platform will then start the GWAS calculation and display the
          status of the job. Once the job is complete, you can download the
          results by clicking the &ldquo;Download Results&rdquo; button. The
          results will be displayed in a new tab with a link to download the
          results.
        </p>
        <br />
        <h2 className="text-xl font-semibold mb-4">
          What is an approximate GWAS?
        </h2>
        <p>
          An approximate GWAS is a method for performing a genome-wide
          association study (GWAS) that is significantly faster than traditional
          GWAS methods. Instead of running a direct GWAS on the phenotype of
          interest, our method approximates that phenotype using other
          phenotypes, then approximate the GWAS summary statistics for the
          desired phenotype using that approximation.
        </p>
        <br />
        <h2 className="text-xl font-semibold mb-4">How does WebGWAS work?</h2>
        <p>
          WebGWAS uses a regression and Indirect GWAS to perform approximate
          GWAS. It involves the following steps:
        </p>
        <br />
        <ol className="list-decimal list-inside">
          <li>
            Build the phenotype of interest using anonymized data from the
            cohort of interest
          </li>
          <li>Approximate the phenotype of interest using other phenotypes</li>
          <li>
            Run an Indirect GWAS to obtain the summary statistics for the
            approximated phenotype
          </li>
        </ol>
        <br />
        <p>
          WebGWAS is designed to be highly efficient and fast, with the ability
          to process large datasets and perform multiple GWAS calculations
          simultaneously. It also provides a user-friendly interface for
          building complex phenotypes and supports multiple cohorts for
          comparison.
        </p>
        <br />
        <h2 className="text-xl font-semibold mb-4">How do I cite WebGWAS?</h2>
        <p>
          If you use WebGWAS in your research, please cite the following paper:
        </p>
        <blockquote className="prose prose-jet">
          <p>TODO</p>
        </blockquote>
        <br />
        <p>You can also use the following BibTeX entry:</p>
        <pre className="bg-gray-100 p-4 rounded-lg">TODO</pre>
      </div>
    </div>
  );
}
