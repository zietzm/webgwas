import React, { useState, useEffect } from "react";
import {
  Play,
  CheckCircle,
  Loader,
  XCircle,
  MinusCircle,
  Download,
} from "lucide-react";
import FuzzySelect from "./FuzzySelect";
import { Cohort, Feature, ListNode, PhenotypeSummary } from "../lib/types";
import Image from "next/image";
import {
  fetchCohorts,
  convertListToRPN,
  runGWAS,
  getResults,
  fetchFeatures,
  getPhenotypeSummary,
  getPvalues,
  PvaluesResult,
} from "../lib/api";
import PhenotypeScatterPlots from "./PhenotypeSummary";
import ManhattanPlot from "./ManhattanPlot";

const API_URL: string = process.env.NEXT_PUBLIC_API_URL || "";

export function simplePhenotypeBuilderUsage() {
  return (
    <div className="p-5 border border-t-0 border-gray-200 dark:border-gray-700">
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        To get started, select a cohort from the buttons below. Then define the
        phenotype that interests you by searching for fields in the search box
        below.
      </p>
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        As an example, suppose we are interested hypertensive diabetes (diabetes
        and hypertension). To build this phenotype, we would add both diabetes
        (E11) and hypertension (I10) as children.
      </p>
      <p className="mb-2 text-gray-600 dark:text-gray-400">
        The resulting phenotype would look like this:
      </p>
      <Image
        src={"/exampleSimplePt.png"}
        alt={"Example phenotype"}
        width="0"
        height="0"
        sizes="100vw"
        className="w-1/2 h-auto"
      />
      <p className="my-2 text-gray-600 dark:text-gray-400">
        Once built, you can validate your phenotype and run the GWAS. Our server
        will then start the GWAS calculation and display the status of the job.
        Once complete, a download link to your results will appear.
      </p>
    </div>
  );
}

export default function SimplePhenotypeBuilder() {
  const [cohorts, setCohorts] = useState<Cohort[]>([]);
  const [selectedCohort, setSelectedCohort] = useState<Cohort | null>(null);
  const [features, setFeatures] = useState<Feature[]>([]);
  const [phenotype, setPhenotype] = useState<ListNode[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState("");
  const [summary, setSummary] = useState<PhenotypeSummary | null>(null);
  const [jobStatus, setJobStatus] = useState<
    "submitting" | "queued" | "done" | "error" | null
  >(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [pvals, setPvals] = useState<PvaluesResult | null>(null);

  // Fetch cohorts
  useEffect(() => {
    async function doFetch() {
      setIsLoading(true);
      try {
        const cohorts = await fetchCohorts(API_URL);
        setCohorts(cohorts);
        setSelectedCohort(cohorts[0]);
      } catch (error) {
        let errorMessage = "Error fetching cohorts";
        if (error instanceof Error) {
          errorMessage = `Error fetching cohorts: ${error.message}`;
        }
        setError(errorMessage);
      }
      setIsLoading(false);
    }
    doFetch();
  }, []);

  // Fetch features
  useEffect(() => {
    async function handleFetchFeatures() {
      if (selectedCohort) {
        const cohortFeatures = await fetchFeatures(API_URL, selectedCohort);
        const binaryFeatures = cohortFeatures.filter((f) => f.type === "BOOL");
        setFeatures(binaryFeatures);
      }
    }
    handleFetchFeatures();
  }, [selectedCohort]);

  async function handleRunGWAS() {
    if (!selectedCohort) {
      alert("Please select a cohort first.");
      return;
    }
    if (!phenotype.length) {
      alert("Please add at least one node to your phenotype.");
    }
    try {
      setJobStatus("submitting");
      const phenotypeDefinition = convertListToRPN(phenotype);
      const phenotypeSummary = await getPhenotypeSummary(
        API_URL,
        phenotypeDefinition,
        selectedCohort,
      );
      setSummary(phenotypeSummary);
      const result = await runGWAS(
        API_URL,
        phenotypeDefinition,
        selectedCohort,
      );
      setJobStatus(result.status);
      await pollJobStatus(result.request_id);
    } catch (err) {
      let errorMessage = "Error running GWAS";
      if (err instanceof Error) {
        errorMessage = `Error running GWAS: ${err.message}`;
      }
      alert(errorMessage);
      setJobStatus("error");
    }
  }

  async function pollJobStatus(requestId: string) {
    try {
      const result = await getResults(API_URL, requestId);
      switch (result.status) {
        case "done":
          // console.debug("GWAS job done");
          setJobStatus("done");
          if (pvals === null) {
            await downloadPvals(requestId);
          }
          downloadResults(requestId);
          return;
        case "error":
          setJobStatus("error");
          // console.error("GWAS job failed. Please try again.", result);
          break;
        case "uploading":
          // console.debug("Server is uploading results");
          if (pvals === null) {
            await downloadPvals(requestId);
          }
        default:
          // console.debug(`Polled job status and got '${result.status}'`);
          setTimeout(() => pollJobStatus(requestId), 1000); // Poll every second
      }
    } catch (err) {
      // console.error("Error polling job status:", err);
      setJobStatus("error");
    }
  }

  async function downloadResults(requestId: string) {
    try {
      const result = await getResults(API_URL, requestId);
      setDownloadUrl(result.url);
    } catch (err) {
      // console.error("Error downloading results:", err);
      alert("Failed to download results. Please try again.");
      setJobStatus("error");
    }
  }

  async function downloadPvals(requestId: string) {
    try {
      const result = await getPvalues(API_URL, requestId);
      setPvals(result);
    } catch (err) {
      // console.error("Error downloading pvalues:", err);
      alert("Failed to download pvalues. Please try again.");
      setJobStatus("error");
    }
  }

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <Loader className="animate-spin" size={40} />
      </div>
    );
  }
  if (error) {
    return <div className="text-red-600 text-center">{error}</div>;
  }

  function CohortSelector() {
    return (
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-4">Select Cohort</h2>
        <div className="flex flex-wrap gap-2">
          {cohorts.map((cohort) => (
            <button
              key={cohort.id}
              onClick={() => {
                if (jobStatus === null || jobStatus === "done") {
                  setSelectedCohort(cohort);
                  setPhenotype([]);
                  setJobStatus(null);
                  setSummary(null);
                } else {
                  alert("Please wait for the current job to finish.");
                }
              }}
              className={`py-2 px-4 rounded-full transition-colors ${
                selectedCohort === cohort
                  ? "bg-blue-600 text-white"
                  : "bg-gray-200 text-gray-800 hover:bg-gray-300"
              }`}
            >
              {cohort.name}
            </button>
          ))}
        </div>
      </div>
    );
  }

  function PhenotypeBuilderDisplay() {
    return (
      <div className="my-6">
        {phenotype.map((node, index) => (
          <div key={index} className="ml-4 flex-1 flex-row gap-2">
            {index > 0 && <b className="text-blue-600">AND </b>}
            {node.negated && jobStatus === null && (
              <button
                onClick={() => {
                  setPhenotype((prevPhenotype) => {
                    const newPhenotype = [...prevPhenotype];
                    newPhenotype[index].negated = false;
                    return newPhenotype;
                  });
                }}
              >
                <b className="mr-1 p-1 text-red-600 bg-red-100 hover:bg-red-200 active:bg-red-300 rounded">
                  NOT
                </b>
              </button>
            )}
            {node.negated && jobStatus !== null && (
              <b className="mr-1 p-1 text-red-600 bg-red-100 rounded">NOT</b>
            )}
            {node.feature.name} [{node.feature.code}] (N=
            {node.feature.sample_size})
            {jobStatus === null && !node.negated && (
              <button
                onClick={() => {
                  setPhenotype((prevPhenotype) => {
                    const newPhenotype = [...prevPhenotype];
                    newPhenotype[index].negated = true;
                    return newPhenotype;
                  });
                }}
              >
                <a className="ml-5 p-1 text-red-600 bg-red-100 hover:bg-red-200 active:bg-red-300 rounded">
                  Negate
                </a>
              </button>
            )}
            {jobStatus === null && (
              <button
                onClick={() => {
                  setPhenotype((prevPhenotype) => {
                    const newPhenotype = [...prevPhenotype];
                    newPhenotype.splice(index, 1);
                    return newPhenotype;
                  });
                }}
                className="ml-2 p-1 rounded hover:bg-gray-200 active:bg-gray-300"
              >
                <MinusCircle size={20} />
              </button>
            )}
          </div>
        ))}
      </div>
    );
  }

  function JobStatusBoxes() {
    return (
      <div className="flex items-center bg-gray-100 p-2 rounded-lg">
        {(jobStatus === "submitting" || jobStatus === "queued") && (
          <Loader className="animate-spin mr-2" size={20} />
        )}
        {jobStatus === "done" && (
          <CheckCircle className="mr-2 text-green-600" size={20} />
        )}
        {jobStatus === "error" && (
          <XCircle className="mr-2 text-red-600" size={20} />
        )}
        <span className="text-gray-700">
          {jobStatus === "queued" && "GWAS job queued..."}
          {jobStatus === "done" && "GWAS job completed."}
          {jobStatus === "error" && "GWAS job failed. Please try again."}
        </span>
      </div>
    );
  }

  function GWASButtons() {
    return (
      <div className="flex flex-wrap gap-4 mb-6">
        {jobStatus === null && (
          <button
            onClick={() => {
              handleRunGWAS();
            }}
            className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors"
          >
            <Play className="mr-2" size={20} />
            Run GWAS
          </button>
        )}
        {jobStatus && <JobStatusBoxes />}
        {jobStatus && jobStatus === "done" && (
          <div>
            <a href={downloadUrl!} download>
              <button className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg flex items-center transition-colors">
                <Download className="mr-2" size={20} />
                Download Results
              </button>
            </a>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white shadow-md rounded-lg p-6 items-center">
      <CohortSelector />
      {phenotype.length > 0 && <PhenotypeBuilderDisplay />}
      {selectedCohort && jobStatus === null && (
        <div className="my-6">
          <FuzzySelect
            fuseThreshold={0.3} // Higher number is more lenient
            options={features}
            closeMenuOnSelect={false}
            onChange={(selectedOption) => {
              const node: ListNode = {
                feature: selectedOption as Feature,
                negated: false,
              };
              setPhenotype([...phenotype, node]);
            }}
            value={null}
            getOptionLabel={(option) =>
              `${option.name} [${option.code}] (N=${option.sample_size})`
            }
            getOptionValue={(option) => option.code}
            placeholder="Search for a field..."
            className="mb-2"
          />
        </div>
      )}
      {phenotype.length > 0 && <GWASButtons />}
      {summary && <PhenotypeScatterPlots data={summary} />}
      {pvals && <ManhattanPlot data={pvals} />}
    </div>
  );
}
