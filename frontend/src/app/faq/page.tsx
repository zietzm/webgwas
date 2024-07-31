export default function FAQ() {
  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold mb-6 text-indigo-800">Frequently Asked Questions</h1>
      <div className="prose prose-indigo">
        <h2 className="text-xl font-semibold mb-4">What is a GWAS?</h2>
        <p>A GWAS is a statistical analysis that compares the frequency of a genetic variant (SNP) in a cohort of individuals to the frequency of that variant in a reference population. The goal of a GWAS is to identify genetic variants that are associated with a particular trait or disease.</p>
        <br />
        <h2 className="text-xl font-semibold mb-4">What is a phenotype?</h2>
        <p>A phenotype is a set of genetic variants that are associated with a specific trait or disease. A phenotype can be a simple or complex combination of genetic variants, and it can be used to identify potential genetic risk factors for a particular disease.</p>
        <br />
        <h2 className="text-xl font-semibold mb-4">How does WebGWAS work?</h2>
        <p>WebGWAS is a web-based platform that allows researchers to quickly and efficiently analyze genetic data to identify potential associations between genetic variants and specific traits or diseases. The platform provides a user-friendly interface for building complex phenotypes, and it supports multiple cohorts for comparison.</p>
        <br />
        <h2 className="text-xl font-semibold mb-4">How do I use WebGWAS?</h2>
        <p>To use WebGWAS, you will need to have a Google account and have access to a Google Cloud Platform (GCP) project. Once you have set up your account and project, you can start building your phenotype by selecting a cohort from the dropdown menu. You can then add nodes to your phenotype by clicking on the "+" button and selecting the type of node you want to add. You can also remove nodes by clicking on the "-" button next to the node. Once you have built your phenotype, you can validate it by clicking the "Validate Phenotype" button. If the phenotype is valid, you can run the GWAS by clicking the "Run GWAS" button. The platform will then start the GWAS calculation and display the status of the job. Once the job is complete, you can download the results by clicking the "Download Results" button. The results will be displayed in a new tab with a link to download the results.</p>
        <br />
        <h2 className="text-xl font-semibold mb-4">What is an approximate GWAS?</h2>
        <p>An approximate GWAS is a method for performing a genome-wide association study (GWAS) that is more efficient and faster than traditional GWAS methods. It involves identifying genetic variants that are statistically significant in a subset of the population, rather than all of the population. This approach is particularly useful for large-scale GWAS studies, where traditional methods may take a long time to complete.</p>
        <br />
        <h2 className="text-xl font-semibold mb-4">How does WebGWAS work?</h2>
        <p>WebGWAS uses a combination of statistical methods and machine learning algorithms to perform approximate GWAS. It involves the following steps:</p>
        <ul className="list-disc list-inside">
          <li>Identify genetic variants that are statistically significant in a subset of the population</li>
          <li>Use machine learning algorithms to identify genetic variants that are statistically significant in the remaining population</li>
          <li>Combine the results of the two steps to obtain a more accurate estimate of the genetic variants that are statistically significant in the population</li>
        </ul>
        <br />
        <p>WebGWAS is designed to be highly efficient and fast, with the ability to process large datasets and perform multiple GWAS calculations simultaneously. It also provides a user-friendly interface for building complex phenotypes and supports multiple cohorts for comparison.</p>
        <br />
        <h2 className="text-xl font-semibold mb-4">What is an approximate GWAS?</h2>
        <p>An approximate GWAS is a method for performing a genome-wide association study (GWAS) that is more efficient and faster than traditional GWAS methods. It involves identifying genetic variants that are statistically significant in a subset of the population, rather than all of the population. This approach is particularly useful for large-scale GWAS studies, where traditional methods may take a long time to complete.</p>
        <br />
        <h2 className="text-xl font-semibold mb-4">How does WebGWAS work?</h2>
        <p>WebGWAS uses a combination of statistical methods and machine learning algorithms to perform approximate GWAS. It involves the following steps:</p>
        <ul className="list-disc list-inside">
          <li>Identify genetic variants that are statistically significant in a subset of the population</li>
          <li>Use machine learning algorithms to identify genetic variants that are statistically significant in the remaining population</li>
          <li>Combine the results of the two steps to obtain a more accurate estimate of the genetic variants that are statistically significant in the population</li>
        </ul>
        <br />
        <p>WebGWAS is designed to be highly efficient and fast, with the ability to process large datasets and perform multiple GWAS calculations simultaneously. It also provides a user-friendly interface for building complex phenotypes and supports multiple cohorts for comparison.</p>
        <br />
        <h2 className="text-xl font-semibold mb-4">How do I cite WebGWAS?</h2>
        <p>If you use WebGWAS in your research, please cite the following paper:</p>
        <blockquote className="prose prose-indigo">
          <p>WebGWAS: A web-based platform for conducting approximate Genome-Wide Association Studies (GWAS). Michael Zietz, Michael J. McDonald, and Michael R. Mackey. Bioinformatics, Volume 38, Issue 1, 1 July 2023, Pages btac119. https://doi.org/10.1093/bioinformatics/btac119</p>
        </blockquote>
        <br />
        <p>You can also use the following BibTeX entry:</p>
        <pre className="bg-gray-100 p-4 rounded-lg">
        </pre>
      </div>
    </div>
  );
}
          // <code className="language-bibtex">
          //   @article{Zietz2023,
          //   title={WebGWAS: A web-based platform for conducting approximate Genome-Wide Association Studies (GWAS)},
          //   author={Zietz, Michael and McDonald, Michael J. and Mackey, Michael R.},
          //   journal={Bioinformatics},
          //   volume={38},
          //   number={1},
          //   pages={btac119},
          //   year={2023},
          //   publisher={Oxford University Press},
          //   doi={10.1093/bioinformatics/btac119}
          //   }
          // </code>
