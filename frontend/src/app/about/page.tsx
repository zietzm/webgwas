export default function About() {
  return (
    <div className="max-w-3xl mx-auto mt-8">
      <h1 className="text-3xl font-bold mb-6 text-blue-dark">About WebGWAS</h1>
      <div className="prose prose-jet">
        <p className="my-4">
          WebGWAS is an online tool for performing instant GWAS on arbitrary
          phenotypes. Instead of taking weeks or months to acquire data, perform
          careful quality control, and conduct a GWAS, researchers can use
          WebGWAS to get preliminary results in seconds. WebGWAS was developed
          in the{" "}
          <a
            href="https://tatonettilab.org/"
            className="underline text-jet font-medium hover:text-blue-dark visited:text-purple-600"
          >
            Tatonetti Lab
          </a>{" "}
          at{" "}
          <a
            href="https://www.dbmi.columbia.edu/"
            className="underline text-jet font-medium hover:text-blue-dark visited:text-purple-600"
          >
            Columbia University
          </a>{" "}
          and{" "}
          <a
            href="https://www.cedars-sinai.edu/research-education/research/departments-institutes/computational-biomedicine.html"
            className="underline text-jet font-medium hover:text-blue-dark visited:text-purple-600"
          >
            Cedars-Sinai Medical Center
          </a>
          .
        </p>
        <p className="text-xl font-semibold mt-4">How it works</p>
        <p className="mb-4 mt-2">
          To study a particular phenotype (e.g. patients with essential
          hypertension and type 2 diabetes), you can define a phenotype in the
          form of either a list (X and Y and ...) or a tree (AND(X, NOT(Y))).
          WebGWAS applies this phenotype definition to anonymized data on our
          server, then tries to approximate that phenotype using the phenotypes
          for which we have pre-computed GWAS summary statistics. The results of
          that approximation allow us to approximate the GWAS summary statistics
          for the provided phenotype.
        </p>
      </div>
      <h1 className="text-3xl font-bold my-6 text-blue-dark">Citing WebGWAS</h1>
      <div className="border-l-4 border-gray-300 pl-4 bg-gray-50 p-4 rounded-md">
        <p className="italic text-lg">
          Phenotype projections accelerate biobank-scale GWAS
        </p>
        <p className="mt-2">
          Michael Zietz, Undina Gisladottir, Kathleen LaRow Brown, Nicholas P.
          Tatonetti
        </p>
        <p className="mt-2">
          bioRxiv 2023.11.20.567948; doi:
          <a
            href="https://doi.org/10.1101/2023.11.20.567948"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline"
          >
            10.1101/2023.11.20.567948
          </a>
        </p>
      </div>
      <h1 className="text-3xl font-bold my-6 text-blue-dark">
        Frequently asked questions
      </h1>
      <div className="border-l-4 border-blue-dark pl-4 mt-2 mb-4 font-semibold">
        <p>Does WebGWAS pose a privacy risk to study participants?</p>
      </div>
      <p className="mb-4">
        WebGWAS only releases unfiltered GWAS summary statistics, which pose a
        minimal risk privacy risk to study participants. These results are
        regularly{" "}
        <a
          href="https://www.ebi.ac.uk/gwas/"
          className="underline text-jet font-medium hover:text-blue-dark visited:text-purple-600"
        >
          shared publicly
        </a>
        .
      </p>
      <p className="mb-4">
        In addition to unfiltered GWAS summary statistics, our server also
        provides true and approximated phenotype values to visualize the quality
        of approximation behind the results. These values are anonymized and
        randomly sampled, minimizing the privacy risk for study participants.
      </p>
      <p className="mb-4">
        Data are anonymized using a method called maximum distance to the
        average vector (MDAV). This method results in k-anonymity with k=10,
        meaning that every individual&apos;s data are shared with at least 9
        other individuals.
      </p>
      <p className="mb-4">
        Even if our server was hacked and all data stored there were released,
        we believe there is no privacy risk to study participants. The only
        pieces of data stored on are server are anonymized, aggregated phenotype
        data, a phenotypic correlation matrix, and GWAS summary statistics.
      </p>
      <div className="border-l-4 border-blue-dark pl-4 mt-2 mb-4 font-semibold">
        <p>What data are available?</p>
      </div>
      <p className="mb-4">
        Currently, WebGWAS only contains data for a White British cohort from
        the UK Biobank. We are in the process of adding additional cohorts,
        particularly of different ancestries, using data from the All of Us
        research program.
      </p>
      <div className="border-l-4 border-blue-dark pl-4 mt-2 mb-4 font-semibold">
        <p>
          What is the difference between the simple and advanced phenotype
          builders?
        </p>
      </div>
      <p className="mb-4">
        The simple phenotype builder allows users to define a phenotype using
        AND and NOT operators. Because of this, it is limited to binary
        phenotypes. The advanced phenotype builder allows users to define more
        complex phenotypes, including arbitrarily complicated phenotype
        definitions.
      </p>
      <div className="border-l-4 border-blue-dark pl-4 mt-2 mb-4 font-semibold">
        <p>Where can I find more information about WebGWAS?</p>
      </div>
      <p className="mb-4">
        For more information, please open an issue on our{" "}
        <a
          href="https://github.com/tatonetti-lab/webgwas"
          className="underline text-jet font-medium hover:text-blue-dark visited:text-purple-600"
        >
          GitHub repository
        </a>{" "}
        or contact Michael Zietz at{" "}
        <a
          href="mailto:michael.zietz@gmail.com"
          className="underline text-jet font-medium hover:text-blue-dark visited:text-purple-600"
        >
          michael.zietz@gmail.com
        </a>
        .
      </p>
    </div>
  );
}
